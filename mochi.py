
import sys
import logging


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)
import cuda_malloc
import comfy.model_management as mm

sys.stdout.write("Imports ...\n")
sys.stdout.flush()

import os

import torch
import argparse, os, sys, glob
import torch
import numpy as np
from tqdm import tqdm
from PIL import Image
from inferStandalone import DownloadAndLoadMochiModel, get_matching_filename, MochiTextEncode, MochiSampler, \
    MochiLatentPreview, MochiDecode, CLIPLoader


def parse_args():
    desc = "Blah"
    parser = argparse.ArgumentParser("CommandLine arguments for mochi1-preview standalone to generate a video")
    parser.add_argument('--prompt', type=str, help='Text to generate video from.', required=True)
    parser.add_argument('--output_path', type=str, help='File name of the saved video including the path.', required=True)
    parser.add_argument('--seed', type=int, help='Random seed.', required=True)
    parser.add_argument('--promptn', type=str, help='Text to avoid in video.', required=False, default='')
    parser.add_argument('--prompt_strength', type=float, default=1.0, required=False,
                        help='might be used in the future if initial samples are provided')
    parser.add_argument('--promptn_strength', type=float, default=1.0, required=False,
                        help='Negative prompt strength')
    parser.add_argument('--fps', type=int, default=24, required=False, help='Frames Per second in the output video')
    parser.add_argument('--width', type=int, help='Video Width', required=False, default=848)
    parser.add_argument('--height', type=int, help='Video Height', required=False, default=480)
    parser.add_argument('--frames', type=int, help='Number of frames to generate.', required=False, default=163)
    parser.add_argument('--steps', type=int, help='number of inference steps. more means more detail but too much and it looks fake', required=False, default=50)
    parser.add_argument('--cfg', type=float, help='Classifier-free guidance scale. between 1.0 and 10.0', required=False, default=4.5)
    parser.add_argument('--decode_frame_batch_size', type=int, required=False, default=10, help='Lowering this reduces memory consumption but adds glitchiness')
    parser.add_argument('--decode_tile_sample_min_height', type=int, required=False, default=160, help='Vae tiling minimum sample height')
    parser.add_argument('--decode_tile_sample_min_width', type=int, required=False, default=312,
                        help='Vae tiling minimum sample width')
    parser.add_argument('--decode_tile_overlap_factor_height', type=float, required=False, default=0.25,
                        help='Overlap over tiles for blending 1/4th default')
    parser.add_argument('--decode_tile_overlap_factor_width', type=float, required=False, default=0.25,
                        help='Overlap over tiles for blending 1/4th default')
    parser.add_argument('--decode_auto_vae_tile_size', type=bool, required=False, default=True)
    parser.add_argument('--verbose', action='store_true', help='Enable verbose output.')
    # parse user args, then add our non-configurable items to args

    # for testing without command line parameters
    # sys.argv.extend([
    #     '--prompt', 'Shrek eating pizza',
    #     '--output_path', 'C:\\Users\\guest\\Documents\\shrek_eating_a_pizza-mochi.mp4',
    #     '--seed', '8083',
    #     '--promptn', '',
    #     '--prompt_strength', '1.0',
    #     '--fps', '24',
    #     '--width', '848',
    #     '--height', '480',
    #     '--frames', '163',
    #     '--steps', '50',
    #     '--cfg', '4.5',
    #     '--decode_frame_batch_size', '10',
    #     '--decode_tile_sample_min_height', '160',
    #     '--decode_tile_sample_min_width', '312',
    #     '--decode_tile_overlap_factor_height', '0.25',
    #     '--decode_tile_overlap_factor_width', '0.25',
    #     '--decode_auto_vae_tile_size', 'True',
    #     '--verbose'
    # ])

    args = parser.parse_args()

    # you can adjust these depending on what you want to do, I just put them under args because it keeps them together in the file
    args.clippath = './models/clip/t5xxl_fp16.safetensors'
    args.vaepath = './models/vae/mochi/mochi_preview_vae_decoder_bf16.safetensors'
    args.mochipath = './models/mochi/mochi_preview_dit_fp8_e4m3fn.safetensors'
    args.precision = 'fp8_e4m3fn'
    args.attention_mode = 'sdpa'
    args.clip_type = 'sd3'
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args.offload_device = torch.device('cpu')
    args.sampler_control_after_generate = 'fixed'
    args.initial_image = 'Not Used Yet, but, if you run an image through the mochi encode method and then provide that to the mochi-sampler, it might be how initial images work in the future'
    args.prompt_force_offload = False
    args.promptn_force_offload = True
    args.enable_vae_tiling = True


    return args

def is_float(element):
    try:
        float(element)
        return True
    except ValueError:
        return False




def load_model(args):
    mochi_modelnames = filenames = [
        "mochi_preview_dit_fp8_e4m3fn.safetensors",
        "mochi_preview_dit_bf16.safetensors",
        "mochi_preview_dit_GGUF_Q4_0_v2.safetensors",
        "mochi_preview_dit_GGUF_Q8_0.safetensors",
    ]
    vae_modelnames = [
        "mochi_preview_vae_decoder_bf16.safetensors",
    ]
    # dtype = {"fp8_e4m3fn": torch.float8_e4m3fn, "fp8_e4m3fn_fast": torch.float8_e4m3fn, "bf16": torch.bfloat16,
    #         "fp16": torch.float16, "fp32": torch.float32}[args.precision]

    mochi_model_downloader = DownloadAndLoadMochiModel()
    mochi_model_name = get_matching_filename(filenames=mochi_modelnames, user_string=args.mochipath)
    vae_model_name = get_matching_filename(filenames=vae_modelnames, user_string=args.vaepath)
    rmochi_model, rvae_model = mochi_model_downloader.loadmodel(model=mochi_model_name, vae=vae_model_name, precision= args.precision, attention_mode=args.attention_mode, args=args)
    return rmochi_model, rvae_model

def load_t5_embeddings(args, pprompt, pstrength, pforce_offload, pclip):
    mochi_Text_encoder = MochiTextEncode()
    t5_embeds, _ = mochi_Text_encoder.process(clip=pclip, prompt=pprompt, strength=pstrength, force_offload=pforce_offload)
    return t5_embeds

def process_sampler(args, mochi_model_obj, positive_embeddings, negative_embeddings):
    mochi_sampler = MochiSampler()
    mochi_latents = mochi_sampler.process(model=mochi_model_obj, positive=positive_embeddings, negative=negative_embeddings, steps=args.steps, cfg=args.cfg, seed=args.seed, height=args.height, width=args.width, num_frames=args.frames)

    return mochi_latents

def get_latent_previews(args, samples):
    mochi_latent_previewer = MochiLatentPreview()
    latent_images = mochi_latent_previewer.sample(args, samples)

    return latent_images

def get_clip(args):
    cliploader = CLIPLoader()
    clip = cliploader.load_clip(clip_name="t5xxl_fp16.safetensors", type="sd3")
    return clip

def get_decoded_latent_preview(args, mochi_latent_samples, vae_model_obj):
    mochi_latent_previewer = MochiDecode()

    frames = mochi_latent_previewer.decode(vae=vae_model_obj, samples=mochi_latent_samples, enable_vae_tiling=args.enable_vae_tiling,
                                           tile_sample_min_height=args.decode_tile_sample_min_height,
                                           tile_sample_min_width=args.decode_tile_sample_min_width,
                                           tile_overlap_factor_height=args.decode_tile_overlap_factor_height,
                                           tile_overlap_factor_width=args.decode_tile_overlap_factor_width,
                                           auto_tile_size=args.decode_auto_vae_tile_size,
                                           frame_batch_size=args.decode_frame_batch_size, unnormalize=False)
    return frames


if __name__ == '__main__':

    args2=parse_args()



    model_paths_check = [
        args2.clippath,
        args2.vaepath,
        args2.mochipath
    ]

    all_Models_exist = True
    for path in model_paths_check:
        if not os.path.exists(path):
            print(f"Model not found: {path}.  Please download the model from Huggingface before running.")
            all_Models_exist = False

    if all_Models_exist:
        with torch.no_grad():
            # Get Clip
            pclip = get_clip(args2)
            print("Clip Loaded")
            negative_embeddings = None
            # Get Clip Embeddings

            positive_embeddings = load_t5_embeddings(args=args2, pprompt=args2.prompt,
                                                     pstrength=args2.prompt_strength,
                                                     pforce_offload=args2.prompt_force_offload, pclip=pclip)
            print("Positive Clip Embeddings")
            if len(args2.promptn) > 0:
                negative_embeddings = load_t5_embeddings(args=args2, pprompt=args2.promptn,
                                                         pstrength=args2.promptn_strength,
                                                         pforce_offload=args2.promptn_force_offload, pclip=pclip)
            else:
                negative_embeddings = load_t5_embeddings(args=args2, pprompt='', pstrength=args2.promptn_strength,
                                                         pforce_offload=args2.promptn_force_offload, pclip=pclip)
            print("Negative Clip Embeddings")
            print("loading Mochi")
            mochi_model, vae_model = load_model(args2)
            print("mochi loaded")
            print("beginning samples")
            # Models loaded
            mochi_latent_samples = process_sampler(args=args2, mochi_model_obj=mochi_model, positive_embeddings=positive_embeddings, negative_embeddings=negative_embeddings)
            # torch.save(mochi_latent_samples, 'z_tensor.pt')
            # mochi_latent_samples = torch.load('z_tensor.pt')[0]
            latent_previews = get_decoded_latent_preview(args=args2, mochi_latent_samples=mochi_latent_samples[0], vae_model_obj=vae_model)
            frames = latent_previews[0].cpu()
            frames_np = frames.numpy()  # Shape: (num_frames, height, width, 3)
            frames_np = (frames_np * 255).astype(np.uint8)
            import imageio

            # Define the output path and frames per second
            output_path = args2.output_path  # Ensure this is a valid path ending with .mp4 or .avi
            fps = args2.fps  # Adjust as needed

            # Write the video
            imageio.mimwrite(output_path, frames_np, fps=fps)





