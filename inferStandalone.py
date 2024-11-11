import os
# import torch._dynamo
# torch._dynamo.config.suppress_errors = True
from comfy.sd import  CLIP,CLIPType, load_text_encoder_state_dicts
import yaml

import torch
from comfy import folder_paths
import comfy.model_management as mm
from einops import rearrange
from tqdm import tqdm
import logging
from comfy.utils import load_torch_file
import safetensors.torch
import comfy.sd
import comfy.utils

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)

from mochi_preview.t2v_synth_mochi import T2VSynthMochiModel
from mochi_preview.vae.model import Decoder, Encoder, add_fourier_features
from mochi_preview.vae.vae_stats import vae_latents_to_dit_latents, dit_latents_to_vae_latents

from contextlib import nullcontext
try:
    from accelerate import init_empty_weights
    from accelerate.utils import set_module_tensor_to_device
    is_accelerate_available = True
except:
    is_accelerate_available = False
    pass

script_directory = os.path.dirname(os.path.abspath(__file__))

def get_matching_filename(filenames, user_string):
    try:
        # Convert filenames to lowercase for case-insensitive comparison
        filenames_lower = [f.lower() for f in filenames]

        # Extract the base filename from the provided path
        base_filename = os.path.basename(user_string).lower()

        # Check if the base filename matches any in the filenames array (case-insensitive)
        if base_filename in filenames_lower:
            return base_filename
        else:
            return None  # No match found
    except Exception as e:
        return f"Error: {e}"  # Return an error message to indicate something went wrong


def linear_quadratic_schedule(num_steps, threshold_noise, linear_steps=None):
    if linear_steps is None:
        linear_steps = num_steps // 2
    linear_sigma_schedule = [i * threshold_noise / linear_steps for i in range(linear_steps)]
    threshold_noise_step_diff = linear_steps - threshold_noise * num_steps
    quadratic_steps = num_steps - linear_steps
    quadratic_coef = threshold_noise_step_diff / (linear_steps * quadratic_steps ** 2)
    linear_coef = threshold_noise / linear_steps - 2 * threshold_noise_step_diff / (quadratic_steps ** 2)
    const = quadratic_coef * (linear_steps ** 2)
    quadratic_sigma_schedule = [
        quadratic_coef * (i ** 2) + linear_coef * i + const
        for i in range(linear_steps, num_steps)
    ]
    sigma_schedule = linear_sigma_schedule + quadratic_sigma_schedule + [1.0]
    sigma_schedule = [1.0 - x for x in sigma_schedule]
    return sigma_schedule


# torch.compile settings, when connected to the model loader, torch.compile of the selected
# layers is attempted. Requires Triton and torch 2.5.0 is recommended
class MochiSigmaSchedule:
    @classmethod
    def loadmodel(self, num_steps, threshold_noise, denoise, linear_steps=None):
        total_steps = num_steps
        if denoise < 1.0:
            if denoise <= 0.0:
                return ([],)
            total_steps = int(num_steps / denoise)

        sigma_schedule = linear_quadratic_schedule(total_steps, threshold_noise, linear_steps)
        sigma_schedule = sigma_schedule[-(num_steps + 1):]
        sigma_schedule = torch.FloatTensor(sigma_schedule)

        return (sigma_schedule,)

# Downloads and loads the selected Mochi model from Huggingface
class DownloadAndLoadMochiModel:
    @classmethod
    def loadmodel(self, model, vae, precision, attention_mode, args=None, trigger=None, compile_args=None, cublas_ops=False, rms_norm_func="default"):

        device = mm.get_torch_device()
        offload_device = mm.unet_offload_device()
        mm.soft_empty_cache()

        dtype = {"fp8_e4m3fn": torch.float8_e4m3fn, "fp8_e4m3fn_fast": torch.float8_e4m3fn, "bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[precision]
        if "fp8" in precision:
            vae_dtype = torch.bfloat16
        else:
            vae_dtype = dtype

        # Transformer model
        if args is not None:
            model_download_path = args.mochipath #os.path.join(folder_paths.models_dir, 'diffusion_models', 'mochi')
            model_path = model_download_path # os.path.join(model_download_path, model)
        else:
            model_download_path = os.path.join(folder_paths.models_dir, 'diffusion_models', 'mochi')
            model_path = os.path.join(model_download_path, model)

        repo_id = "kijai/Mochi_preview_comfy"

        if not os.path.exists(model_path):
            log.info(f"Downloading mochi model to: {model_path}")
            from huggingface_hub import snapshot_download
            snapshot_download(
                repo_id=repo_id,
                allow_patterns=[f"*{model}*"],
                local_dir=model_download_path,
                local_dir_use_symlinks=False,
            )
        # VAE
        if args is not None:
            vae_download_path = args.vaepath #os.path.join(folder_paths.models_dir, 'vae', 'mochi')
            vae_path = vae_download_path #os.path.join(vae_download_path, vae)
        else:
            vae_download_path = os.path.join(folder_paths.models_dir, 'vae', 'mochi')
            vae_path = os.path.join(vae_download_path, vae)

        if not os.path.exists(vae_path):
            log.info(f"Downloading mochi VAE to: {vae_path}")
            from huggingface_hub import snapshot_download
            snapshot_download(
                repo_id=repo_id,
                allow_patterns=[f"*{vae}*"],
                local_dir=vae_download_path,
                local_dir_use_symlinks=False,
            )

        model = T2VSynthMochiModel(
            device=device,
            offload_device=offload_device,
            # vae_stats_path=os.path.join(script_directory, "configs", "vae_stats.json"),
            dit_checkpoint_path=model_path,
            weight_dtype=dtype,
            fp8_fastmode=True if precision == "fp8_e4m3fn_fast" else False,
            attention_mode=attention_mode,
            rms_norm_func=rms_norm_func,
            compile_args=compile_args,
            cublas_ops=cublas_ops
        )
        with (init_empty_weights() if is_accelerate_available else nullcontext()):
            vae = Decoder(
                out_channels=3,
                base_channels=128,
                channel_multipliers=[1, 2, 4, 6],
                temporal_expansions=[1, 2, 3],
                spatial_expansions=[2, 2, 2],
                num_res_blocks=[3, 3, 4, 6, 3],
                latent_dim=12,
                has_attention=[False, False, False, False, False],
                output_norm=False,
                nonlinearity="silu",
                output_nonlinearity="silu",
                causal=True,
                dtype=vae_dtype,
            )
        vae_sd = load_torch_file(vae_path)
        if is_accelerate_available:
            for key in vae_sd:
                set_module_tensor_to_device(vae, key, dtype=vae_dtype, device=offload_device, value=vae_sd[key])
        else:
            vae.load_state_dict(vae_sd, strict=True)
            vae.eval().to(vae_dtype).to("cpu")
        del vae_sd

        return (model, vae,)

class MochiModelLoader:
    @classmethod
    def loadmodel(self, model_name, precision, attention_mode, trigger=None, compile_args=None, cublas_ops=False, rms_norm_func="default"):

        device = mm.get_torch_device()
        offload_device = mm.unet_offload_device()
        mm.soft_empty_cache()

        dtype = {"fp8_e4m3fn": torch.float8_e4m3fn, "fp8_e4m3fn_fast": torch.float8_e4m3fn, "bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[precision]
        model_path = folder_paths.get_full_path_or_raise("diffusion_models", model_name)

        model = T2VSynthMochiModel(
            device=device,
            offload_device=offload_device,
            vae_stats_path=os.path.join(script_directory, "configs", "vae_stats.json"),
            dit_checkpoint_path=model_path,
            weight_dtype=dtype,
            fp8_fastmode = True if precision == "fp8_e4m3fn_fast" else False,
            attention_mode=attention_mode,
            rms_norm_func=rms_norm_func,
            compile_args=compile_args,
            cublas_ops=cublas_ops
        )

        return (model, )

# torch.compile settings, when connected to the model loader, torch.compile of the selected layers is attempted. Requires Triton and torch 2.5.0 is recommended
class MochiTorchCompileSettings:
    @classmethod
    def loadmodel(self, backend, fullgraph, mode, compile_dit, compile_final_layer, dynamic, dynamo_cache_size_limit):

        compile_args = {
            "backend": backend,
            "fullgraph": fullgraph,
            "mode": mode,
            "compile_dit": compile_dit,
            "compile_final_layer": compile_final_layer,
            "dynamic": dynamic,
            "dynamo_cache_size_limit": dynamo_cache_size_limit,
        }

        return (compile_args, )

class MochiVAELoader:
    @classmethod
    def loadmodel(self, model_name, torch_compile_args=None, precision="bf16"):

        device = mm.get_torch_device()
        offload_device = mm.unet_offload_device()
        mm.soft_empty_cache()

        dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[precision]

        vae_path = folder_paths.get_full_path_or_raise("vae", model_name)

        with (init_empty_weights() if is_accelerate_available else nullcontext()):
            vae = Decoder(
                out_channels=3,
                base_channels=128,
                channel_multipliers=[1, 2, 4, 6],
                temporal_expansions=[1, 2, 3],
                spatial_expansions=[2, 2, 2],
                num_res_blocks=[3, 3, 4, 6, 3],
                latent_dim=12,
                has_attention=[False, False, False, False, False],
                output_norm=False,
                nonlinearity="silu",
                output_nonlinearity="silu",
                causal=True,
                dtype=dtype,
            )
        vae_sd = load_torch_file(vae_path)
        if is_accelerate_available:
            for name, param in vae.named_parameters():
                set_module_tensor_to_device(vae, name, dtype=dtype, device=offload_device, value=vae_sd[name])
        else:
            vae.load_state_dict(vae_sd, strict=True)
            vae.to(dtype).to(offload_device)
        vae.eval()
        del vae_sd

        if torch_compile_args is not None:
            vae.to(device)
            vae = torch.compile(vae, fullgraph=torch_compile_args["fullgraph"], mode=torch_compile_args["mode"], dynamic=False, backend=torch_compile_args["backend"])

        return (vae,)

class MochiVAEEncoderLoader:
    @classmethod
    def loadmodel(self, model_name, torch_compile_args=None, precision="bf16"):

        device = mm.get_torch_device()
        offload_device = mm.unet_offload_device()
        mm.soft_empty_cache()

        dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[precision]

        config = dict(
            prune_bottlenecks=[False, False, False, False, False],
            has_attentions=[False, True, True, True, True],
            affine=True,
            bias=True,
            input_is_conv_1x1=True,
            padding_mode="replicate"
        )

        vae_path = folder_paths.get_full_path_or_raise("vae", model_name)

        # Create VAE encoder
        with (init_empty_weights() if is_accelerate_available else nullcontext()):
            encoder = Encoder(
                in_channels=15,
                base_channels=64,
                channel_multipliers=[1, 2, 4, 6],
                num_res_blocks=[3, 3, 4, 6, 3],
                latent_dim=12,
                temporal_reductions=[1, 2, 3],
                spatial_reductions=[2, 2, 2],
                dtype=dtype,
                **config,
            )

        encoder_sd = load_torch_file(vae_path)
        if is_accelerate_available:
            for name, param in encoder.named_parameters():
                set_module_tensor_to_device(encoder, name, dtype=dtype, device=offload_device, value=encoder_sd[name])
        else:
            encoder.load_state_dict(encoder_sd, strict=True)
            encoder.to(dtype).to(offload_device)
        encoder.eval()
        del encoder_sd

        if torch_compile_args is not None:
            encoder.to(device)
            encoder = torch.compile(encoder, fullgraph=torch_compile_args["fullgraph"], mode=torch_compile_args["mode"], dynamic=False, backend=torch_compile_args["backend"])

        return (encoder,)


class MochiTextEncode:
    @classmethod
    def process(self, clip, prompt, strength=1.0, force_offload=True):
        max_tokens = 256
        load_device = mm.text_encoder_device()
        offload_device = mm.text_encoder_offload_device()

        try:
            clip.tokenizer.t5xxl.pad_to_max_length = True
            clip.tokenizer.t5xxl.max_length = max_tokens
            clip.cond_stage_model.t5xxl.return_attention_masks = True
            clip.cond_stage_model.t5xxl.enable_attention_masks = True
            clip.cond_stage_model.t5_attention_mask = True
            clip.cond_stage_model.to(load_device)
            tokens = clip.tokenizer.t5xxl.tokenize_with_weights(prompt, return_word_ids=True)
            try:
                embeds, _, attention_mask = clip.cond_stage_model.t5xxl.encode_token_weights(tokens)
            except:
                NotImplementedError("Failed to get attention mask from T5, is your ComfyUI up to date?")
        except:
            clip.cond_stage_model.to(offload_device)
            tokens = clip.tokenizer.tokenize_with_weights(prompt, return_word_ids=True)
            embeds, _, attention_mask = clip.cond_stage_model.encode_token_weights(tokens)

        if embeds.shape[1] > 256:
            raise ValueError(f"Prompt is too long, max tokens supported is {max_tokens} or less, got {embeds.shape[1]}")
        embeds *= strength
        if force_offload:
            clip.cond_stage_model.to(offload_device)
            mm.soft_empty_cache()

        t5_embeds = {
            "embeds": embeds,
            "attention_mask": attention_mask["attention_mask"].bool(),
        }
        return (t5_embeds, clip,)
class MochiFasterCache:
    def args(self, start_step, hf_step, lf_step, cache_device):
        device = mm.get_torch_device()
        offload_device = mm.unet_offload_device()
        fastercache = {
            "start_step" : start_step,
            "hf_step" : hf_step,
            "lf_step" : lf_step,
            "cache_device" : device if cache_device == "main_device" else offload_device
        }
        return (fastercache,)

class MochiSampler:
    @classmethod
    def process(self, model, positive, negative, steps, cfg, seed, height, width, num_frames, cfg_schedule=None,
                opt_sigmas=None, samples=None, fastercache=None):
        mm.unload_all_models()
        mm.soft_empty_cache()

        if opt_sigmas is not None:
            sigma_schedule = opt_sigmas.tolist()
            steps = int(len(sigma_schedule))
            sigma_schedule.extend([0.0])
            logging.info(f"Using sigma_schedule: {sigma_schedule}")
        else:
            sigma_schedule = linear_quadratic_schedule(steps, 0.025)

        if cfg_schedule is None:
            cfg_schedule = [cfg] * steps
        else:
            logging.info(f"Using cfg schedule: {cfg_schedule}")

        # For compatibility with Comfy CLIPTextEncode
        if not isinstance(positive, dict):
            positive = {
                "embeds": positive[0][0],
                "attention_mask": positive[0][1]["attention_mask"].bool(),
            }
        if not isinstance(negative, dict):
            negative = {
                "embeds": negative[0][0],
                "attention_mask": negative[0][1]["attention_mask"].bool(),
            }

        args = {
            "height": height,
            "width": width,
            "num_frames": num_frames,
            "mochi_args": {
                "sigma_schedule": sigma_schedule,
                "cfg_schedule": cfg_schedule,
                "num_inference_steps": steps,
            },
            "positive_embeds": positive,
            "negative_embeds": negative,
            "seed": seed,
            "samples": samples["samples"] if samples is not None else None,
            "fastercache": fastercache
        }
        latents = model.run(args)

        mm.soft_empty_cache()

        return ({"samples": latents},)

class MochiDecode:
    @classmethod

    def decode(self, vae, samples, enable_vae_tiling, tile_sample_min_height, tile_sample_min_width, tile_overlap_factor_height, 
               tile_overlap_factor_width, auto_tile_size, frame_batch_size, unnormalize=False):
        with torch.no_grad():
            device = mm.get_torch_device()
            offload_device = mm.unet_offload_device()
            intermediate_device = mm.intermediate_device()
            samples = samples["samples"]
            if unnormalize:
                samples = dit_latents_to_vae_latents(samples)
            samples = samples.to(vae.dtype).to(device)

            B, C, T, H, W = samples.shape

            self.tile_overlap_factor_height = tile_overlap_factor_height if not auto_tile_size else 1 / 6
            self.tile_overlap_factor_width = tile_overlap_factor_width if not auto_tile_size else 1 / 5

            self.tile_sample_min_height = tile_sample_min_height if not auto_tile_size else H // 2 * 8
            self.tile_sample_min_width = tile_sample_min_width if not auto_tile_size else W // 2 * 8

            self.tile_latent_min_height = int(self.tile_sample_min_height / 8)
            self.tile_latent_min_width = int(self.tile_sample_min_width / 8)

        def blend_v(a: torch.Tensor, b: torch.Tensor, blend_extent: int) -> torch.Tensor:
            blend_extent = min(a.shape[3], b.shape[3], blend_extent)
            for y in range(blend_extent):
                b[:, :, :, y, :] = a[:, :, :, -blend_extent + y, :] * (1 - y / blend_extent) + b[:, :, :, y, :] * (
                        y / blend_extent
                )
            return b

        def blend_h(a: torch.Tensor, b: torch.Tensor, blend_extent: int) -> torch.Tensor:
            blend_extent = min(a.shape[4], b.shape[4], blend_extent)
            for x in range(blend_extent):
                b[:, :, :, :, x] = a[:, :, :, :, -blend_extent + x] * (1 - x / blend_extent) + b[:, :, :, :, x] * (
                        x / blend_extent
                )
            return b

        def decode_tiled(samples):

            with torch.no_grad():
                overlap_height = int(self.tile_latent_min_height * (1 - self.tile_overlap_factor_height))
                overlap_width = int(self.tile_latent_min_width * (1 - self.tile_overlap_factor_width))
                blend_extent_height = int(self.tile_sample_min_height * self.tile_overlap_factor_height)
                blend_extent_width = int(self.tile_sample_min_width * self.tile_overlap_factor_width)
                row_limit_height = self.tile_sample_min_height - blend_extent_height
                row_limit_width = self.tile_sample_min_width - blend_extent_width

                # Split z into overlapping tiles and decode them separately.
                # The tiles have an overlap to avoid seams between tiles.
                # comfy_pbar = ProgressBar(len(range(0, H, overlap_height)))
                rows = []
                for i in tqdm(range(0, H, overlap_height), desc="Processing rows"):
                    row = []
                    for j in tqdm(range(0, W, overlap_width), desc="Processing columns", leave=False):
                        time = []
                        for k in tqdm(range(T // frame_batch_size), desc="Processing frames", leave=False):
                            remaining_frames = T % frame_batch_size
                            start_frame = frame_batch_size * k + (0 if k == 0 else remaining_frames)
                            end_frame = frame_batch_size * (k + 1) + remaining_frames
                            tile = samples[
                                   :,
                                   :,
                                   start_frame:end_frame,
                                   i: i + self.tile_latent_min_height,
                                   j: j + self.tile_latent_min_width,
                                   ]
                            tile = vae(tile)
                            time.append(tile)
                        row.append(torch.cat(time, dim=2))
                    rows.append(row)
                    # comfy_pbar.update(1)

                result_rows = []
                for i, row in enumerate(tqdm(rows, desc="Blending rows")):
                    result_row = []
                    for j, tile in enumerate(tqdm(row, desc="Blending tiles", leave=False)):
                        # blend the above tile and the left tile
                        # to the current tile and add the current tile to the result row
                        if i > 0:
                            tile = blend_v(rows[i - 1][j], tile, blend_extent_height)
                        if j > 0:
                            tile = blend_h(row[j - 1], tile, blend_extent_width)
                        result_row.append(tile[:, :, :, :row_limit_height, :row_limit_width])
                    result_rows.append(torch.cat(result_row, dim=4))

            return torch.cat(result_rows, dim=3)

        vae.to(device)
        with torch.autocast(mm.get_autocast_device(device), dtype=vae.dtype):
            if enable_vae_tiling and frame_batch_size > T:
                logging.warning(f"Frame batch size is larger than the number of samples, setting to {T}")
                frame_batch_size = T
                frames = decode_tiled(samples)
            elif not enable_vae_tiling:
                logging.warning("Attempting to decode without tiling, very memory intensive")
                frames = vae(samples)
            else:
                logging.info("Decoding with tiling")
                frames = decode_tiled(samples)

        vae.to(offload_device)

        frames = frames.float()
        frames = (frames + 1.0) / 2.0
        frames.clamp_(0.0, 1.0)

        frames = rearrange(frames, "b c t h w -> (t b) h w c").to(intermediate_device)

        return (frames,)

class MochiDecodeSpatialTiling:
    @classmethod
    def decode(self, vae, samples, enable_vae_tiling, num_tiles_w, num_tiles_h, overlap,
               min_block_size, per_batch, unnormalize=True):
        device = mm.get_torch_device()
        offload_device = mm.unet_offload_device()
        intermediate_device = mm.intermediate_device()
        samples = samples["samples"]
        if unnormalize:
        	samples = dit_latents_to_vae_latents(samples)
        samples = samples.to(vae.dtype).to(device)

        B, C, T, H, W = samples.shape

        vae.to(device)
        decoded_list = []
        with torch.no_grad():
            with torch.autocast(mm.get_autocast_device(device), dtype=vae.dtype):
                if enable_vae_tiling:
                    from .mochi_preview.vae.model import apply_tiled

                    # pbar = ProgressBar(T // per_batch)
                    for i in range(0, T, per_batch):
                        if i >= T:
                            break
                        end_index = min(i + per_batch, T)
                        logging.info(f"Decoding {end_index - i} samples with tiling...")
                        chunk = samples[:, :, i:end_index, :, :]
                        frames = apply_tiled(vae, chunk, num_tiles_w = num_tiles_w, num_tiles_h = num_tiles_h, overlap=overlap, min_block_size=min_block_size)
                        logging.info(f"Decoded {frames.shape[2]} frames from {end_index - i} samples")
                        # pbar.update(1)
                        # Blend the first and last frames of each pair
                        if len(decoded_list) > 0:
                            previous_frames = decoded_list[-1]
                            blended_frames = (previous_frames[:, :, -1:, :, :] + frames[:, :, :1, :, :]) / 2
                            decoded_list[-1][:, :, -1:, :, :] = blended_frames

                        decoded_list.append(frames)
                    frames = torch.cat(decoded_list, dim=2)
                else:
                    logging.info("Decoding without tiling...")
                    frames = vae(samples)

        vae.to(offload_device)

        frames = frames.float()
        frames = (frames + 1.0) / 2.0
        frames.clamp_(0.0, 1.0)

        frames = rearrange(frames, "b c t h w -> (t b) h w c").to(intermediate_device)

        return (frames,)

class MochiImageEncode:
    @classmethod
    def decode(self, encoder, images, enable_vae_tiling, num_tiles_w, num_tiles_h, overlap, min_block_size):
        device = mm.get_torch_device()
        offload_device = mm.unet_offload_device()
        intermediate_device = mm.intermediate_device()
        from .mochi_preview.vae.model import apply_tiled
        B, H, W, C = images.shape

        import torchvision.transforms as transforms
        normalize = transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        input_image_tensor = rearrange(images, 'b h w c -> b c h w')
        input_image_tensor = normalize(input_image_tensor).unsqueeze(0)
        input_image_tensor = rearrange(input_image_tensor, 'b t c h w -> b c t h w', t=B)
        
        #images = images.unsqueeze(0).sub_(0.5).div_(0.5)
        #images = rearrange(input_image_tensor, "b c t h w -> t c b h w")
        images = input_image_tensor.to(device)
        
        encoder.to(device)
        print("images before encoding", images.shape)
        with torch.autocast(mm.get_autocast_device(device), dtype=encoder.dtype):
            video = add_fourier_features(images)
            if enable_vae_tiling:
                    latents = apply_tiled(encoder, video, num_tiles_w = num_tiles_w, num_tiles_h = num_tiles_h, overlap=overlap, min_block_size=min_block_size)
            else:
                latents = encoder(video)
        if normalize:
        	latents = vae_latents_to_dit_latents(latents)
        print("encoder output", latents.shape)

        return ({"samples": latents},)

class MochiLatentPreview:
    @classmethod
    def sample(self, samples):  # , seed, min_val, max_val):
        mm.soft_empty_cache()

        latents = samples["samples"].clone()
        print("in sample", latents.shape)

        device = mm.get_torch_device()
        offload_device = mm.unet_offload_device()

        latent_rgb_factors = [[0.1236769792512748, 0.11775175335219157, -0.17700629766423637], [-0.08504104329270078, 0.026605813147523694, -0.006843165704926019], [-0.17093308616366876, 0.027991854696200386, 0.14179146288816308], [-0.17179555328757623, 0.09844317368603078, 0.14470997015982784], [-0.16975067171668484, -0.10739852629856643, -0.1894254942909962], [-0.19315259266769888, -0.011029760569485209, -0.08519702054654255], [-0.08399895091432583, -0.0964246452052032, -0.033622359523655665], [0.08148916330842498, 0.027500645903400067, -0.06593099749891196], [0.0456603103902293, -0.17844808072462398, 0.04204775167149785], [0.001751626383204502, -0.030567890189647867, -0.022078082809772193], [0.05110631095056278, -0.0709677393548804, 0.08963683539504264], [0.010515800868829, -0.18382052841762514, -0.08554553339721907]]

        # import random
        # random.seed(seed)
        # latent_rgb_factors = [[random.uniform(min_val, max_val) for _ in range(3)] for _ in range(12)]
        # out_factors = latent_rgb_factors
        # print(latent_rgb_factors)

        latent_rgb_factors_bias = [0, 0, 0]

        latent_rgb_factors = torch.tensor(latent_rgb_factors, device=latents.device, dtype=latents.dtype).transpose(0, 1)
        latent_rgb_factors_bias = torch.tensor(latent_rgb_factors_bias, device=latents.device, dtype=latents.dtype)

        print("latent_rgb_factors", latent_rgb_factors.shape)

        latent_images = []
        for t in range(latents.shape[2]):
            latent = latents[:, :, t, :, :]
            latent = latent[0].permute(1, 2, 0)
            latent_image = torch.nn.functional.linear(
                latent,
                latent_rgb_factors,
                bias=latent_rgb_factors_bias
            )
            latent_images.append(latent_image)
        latent_images = torch.stack(latent_images, dim=0)
        print("latent_images", latent_images.shape)
        latent_images_min = latent_images.min()
        latent_images_max = latent_images.max()
        latent_images = (latent_images - latent_images_min) / (latent_images_max - latent_images_min)

        return (latent_images.float().cpu(),)
class CLIPLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "clip_name": (folder_paths.get_filename_list("clip"), ),
                              "type": (["stable_diffusion", "stable_cascade", "sd3", "stable_audio", "mochi"], ),
                             }}
    RETURN_TYPES = ("CLIP",)
    FUNCTION = "load_clip"

    CATEGORY = "advanced/loaders"

    def load_clip(self, clip_name, type="sd3"):
        if type == "stable_cascade":
            clip_type = comfy.sd.CLIPType.STABLE_CASCADE
        elif type == "sd3":
            clip_type = comfy.sd.CLIPType.SD3
        elif type == "stable_audio":
            clip_type = comfy.sd.CLIPType.STABLE_AUDIO
        elif type == "mochi":
            clip_type = comfy.sd.CLIPType.MOCHI
        else:
            clip_type = comfy.sd.CLIPType.STABLE_DIFFUSION

        clip_path = folder_paths.get_full_path_or_raise("clip", clip_name)
        print(f"Clip path: {clip_path}")
        print(f"Embedding directory: {folder_paths.get_folder_paths('embeddings')}")
        clip = comfy.sd.load_clip(ckpt_paths=[clip_path], embedding_directory=folder_paths.get_folder_paths("embeddings"), clip_type=clip_type)

        return clip


 #  def load_clip(self, clip_name, type="stable_diffusion"):
 #       # Determine clip type
 #       clip_type_mapping = {
 #           "stable_cascade": CLIPType.STABLE_CASCADE,
 #           "sd3": CLIPType.SD3,
 #           "stable_audio": CLIPType.STABLE_AUDIO,
 #           "mochi": CLIPType.MOCHI,
 #       }
 #       clip_type = clip_type_mapping.get(type, CLIPType.STABLE_DIFFUSION)
#
#        # Load clip state dict
#        clip_path = folder_paths.get_full_path_or_raise("clip", clip_name)
#        clip_data = load_torch_file(clip_path)
#
#        # Define an empty class for clip target with the necessary attributes
#        class ClipTarget:
#            pass
#
#        clip_target = ClipTarget()
#        # Load the parameters from the configuration file if provided
#        if config_path is not None:
#            with open(config_path, 'r') as stream:
#                try:
#                    params = yaml.safe_load(stream)
#                except yaml.YAMLError as exc:
#                    raise ValueError(f"Error loading YAML configuration file: {exc}")
#        else:
#            params = {}
#
#        # Assign loaded params
#        clip_target.params = params
#        # Load text encoder model from clip data
#        te_model = load_text_encoder_state_dicts([clip_data], clip_type=clip_type)
#
#        # If te_model exists, set clip and tokenizer in clip_target
#        if te_model is not None:
#            clip_target.clip = te_model.clip
#            clip_target.tokenizer = te_model.tokenizer
#
#        # Create a new CLIP instance using clip_target
#        clip = CLIP(target=clip_target)
#        return (clip,)
