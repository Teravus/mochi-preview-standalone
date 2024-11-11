# This is a standalone script for the ComfyUI impelementation of the ComfyUI wrapper nodes for [Mochi](https://github.com/genmoai/models) video generator

This repo is for a very specific purpose and you should probably use the ComfyUI wrapper version because it will be more up-to-date and you won't have to fully reload the models every run of the PHP script.

# kijai is still working on this for comfyUI and you should really use that implementation unless you know what you're doing.


Can use flash_attn, pytorch attention (sdpa) or [sage attention](https://github.com/thu-ml/SageAttention), sage being fastest.

Depending on frame count can fit under 20GB, VAE decoding is heavy and there is experimental tiled decoder (taken from CogVideoX -diffusers code) which allows higher frame counts, so far highest I've done is 97 with the default tile size 2x2 grid.

This uses Kijai models and the Flux Clip t5xxl_fp16.safetensors model
Models:

https://huggingface.co/Kijai/Mochi_preview_comfy/tree/main

[mochi_preview_dit_fp8_e4m3fn.safetensors](https://huggingface.co/Kijai/Mochi_preview_comfy/resolve/main/mochi_preview_dit_fp8_e4m3fn.safetensors) model to: `models/mochi`

[mochi_preview_vae_decoder_bf16.safetensors](https://huggingface.co/Kijai/Mochi_preview_comfy/resolve/main/mochi_preview_vae_decoder_bf16.safetensors) vae model to: `models/vae/mochi`

Flux Clip T5xxl model:
[t5xxl_fp16.safetensors](https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/t5xxl_fp16.safetensors) to `models/clip`


There's sort of a partial autodownloader module for some of these models, but it often leads to reasources on huggingface that are there sometimes, other times they're not. 
You should probably get the models directly from huggingface first rather than rely on the autodownloader.

I included third party licenses in the 3rd party licenses folder.  Again, this uses code from ComfyUI, from genmoa-mochi, and from kijai.
Because this uses code from ComfyUI, this repo must be licensed GPLv3.  

You may take the work that I've done under Apache 2, but, Clip loading, for example is done with code 
using ComfyUI. Some dependencies for Kijai's work uses reasources in ComfyUI. 
Memory management including direct allocation is ComfyUI. 

The GPLv3 limits my rights as to how to license the resulting product. 
Some people think GPL gives people freedom.  My opinion, is it actually takes away freedom. 

This isn't about hating on the GPL, just me feeling like I have to explain the rationalle for 
licensing this that way..  since I don't like it.  I would rather license this as Apache 2 
since mochi and kijai's work is licensed Apache 2 but GPL limits my rights due to ComfyUI's 
choice of license and my choice to try and preserve the workflow so maintenance and keeping 
up with kijai will be easier.  This is still a moving target.

# Setup 

I used pip freeze to generate the requirements.txt  Hopefully it will work for you but...  at the end of the day, follow the comfyui install directions and add the package einops and you'll be close-enough.
  https://github.com/comfyanonymous/ComfyUI/ 
Again, this is a very specific purpose package and isn't packaged for easy use.

# Command Line Usage

CommandLine arguments for mochi.py to generate a video
 --prompt PROMPT 
 --output_path OUTPUT_path
 --seed SEED 
 [--promptn PROMPTN]
 [--prompt_strength PROMPT_STRENGTH]
 [--promptn_strength PROMPTN_STRENGTH]
 [--fps FPS] 
 [--width WIDTH] 
 [--height HEIGHT]
 [--frames FRAMES] 
 [--steps STEPS] 
 [--cfg CFG]
 [--decode_frame_batch_size DECODE_FRAME_BATCH_SIZE]
 [--decode_tile_sample_min_height DECODE_TILE_SAMPLE_MIN_HEIGHT]
 [--decode_tile_sample_min_width DECODE_TILE_SAMPLE_MIN_WIDTH]
 [--decode_tile_overlap_factor_height DECODE_TILE_OVERLAP_FACTOR_HEIGHT]
 [--decode_tile_overlap_factor_width DECODE_TILE_OVERLAP_FACTOR_WIDTH]
 [--decode_auto_vae_tile_size DECODE_AUTO_VAE_TILE_SIZE]
 
 * prompt - This is the text prompt for the video that you want to generate.
 * output_path - This is the full path and file name of the output video.
 * seed - This is the random seed used. If you provide 0, it will be random.
 * prompt_strength - this is the strength of the prompt from 1 to 0.
 * fps - The frames per second of the resulting video
 * width - the output width of the video. Note that Mochi uses a smaller representation of the latents to keep the memory usage down
 * height - the height of the video output.
 * frames - The number of frames that a video should have.  More increases the memory usage. Must be divisable by six.
 * steps - More steps makes things clearer to a point.  After that point they look hard edged and hard broiled.  50 works reasonably
 * cfg - from 0 to 10 is reasonable. Balances between generating diverse outputs and strongly adhering to the prompt.
 
 Vae decoding is heavy so it is tiled. 
 You have the ability to control this tiling with the rest of the command line options.  Use the defaults as a general guide but some people may prefer different settings. Some people had better results with width tiling 4, height tiling 4, overlap 16, min-block size 1.
 
 Example Simple usage;
 python mochi.py --prompt "Shrek eating a pizza" --output_path "C:\\users\guest\Documents\Shrek_eating_a_pizza-mochi.mp4" --seed 0
 
