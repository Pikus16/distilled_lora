from diffusers import StableDiffusionXLPipeline, AutoencoderKL, AutoPipelineForText2Image
import torch
from PIL import Image
import numpy as np
import os
from tqdm import tqdm
import json
from diffusers.utils import make_image_grid
import pandas as pd
import diffusers
from huggingface_hub import hf_hub_download
import click

DEFAULT_LORA_MODELS = ['PE_BalloonStyle', 'tintinia', 'pixel-art-xl', 'crayons_v1_sdxl', 'ColoringBookAF', 'v5lcn', 'papercut']
DEFAULT_N_STEPS = [1, 3, 5, 10]

def load_pipe(model_name, lora_path):
    pipe_ = AutoPipelineForText2Image.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        use_safetensors=True,
    ).to("cuda")
    pipe_.load_lora_weights(lora_path)
    pipe_.set_progress_bar_config(disable=True)
    diffusers.logging.set_verbosity_error()
    return pipe_

def generate_images(pipe, fast_pipe, prompt_list, dst_dir, n_steps_list):
    os.makedirs(dst_dir, exist_ok=True)
    # generate standard images
    regular_dir = f'{dst_dir}/regular'
    os.makedirs(regular_dir, exist_ok=True)
    for i, prompt in tqdm(enumerate(prompt_list), total=len(prompt_list)):
        images = pipe(prompt, generator=torch.Generator(device="cuda").manual_seed(0)).images
        images[0].save(f'{regular_dir}/img_{i}.png')

    # generate fast images
    for n_steps in n_steps_list:
        fast_dir = f'{dst_dir}/fast_{n_steps}'
        os.makedirs(fast_dir, exist_ok=True)
        for i, prompt in tqdm(enumerate(prompt_list), total=len(prompt_list)):
            images = fast_pipe(prompt=prompt, num_inference_steps=n_steps, guidance_scale=0.0, generator=torch.Generator(device="cuda").manual_seed(0)).images
            images[0].save(f'{fast_dir}/img_{i}.png')
    
def clear_pipes(pipe, fast_pipe):
    pipe.to('cpu')
    fast_pipe.to('cpu')
    del pipe
    del fast_pipe
    torch.cuda.empty_cache()

@click.command()
@click.option('--lora-models', '-l', multiple=True, default=DEFAULT_LORA_MODELS,
              help='LoRA models to use. Can be specified multiple times. Default: all default models.')
@click.option('--n-steps', '-n', multiple=True, type=int, default=DEFAULT_N_STEPS,
              help='Number of inference steps for fast generation. Can be specified multiple times. Default: 1,3,5,10.')
def main(lora_models, n_steps):
    """Generate images using specified LoRA models and inference steps."""
    # If no models are specified, use the defaults
    if not lora_models:
        lora_models = DEFAULT_LORA_MODELS
    
    # If no steps are specified, use the defaults
    if not n_steps:
        n_steps = DEFAULT_N_STEPS
    
    for lora_base in lora_models:
        print(f"Processing LoRA model: {lora_base}")
        lora_path = f'loras/{lora_base}.safetensors'
        prompt_path = f'prompts/{lora_base}.json'
        
        try:
            with open(prompt_path, 'r') as f:
                prompts = json.load(f)
        except FileNotFoundError:
            print(f"Warning: Prompt file not found at {prompt_path}. Skipping this model.")
            continue
        
        pipe = load_pipe('stabilityai/stable-diffusion-xl-base-1.0', lora_path)
        fast_pipe = load_pipe('stabilityai/sdxl-turbo', lora_path)
        generate_images(pipe, fast_pipe, prompts, f'images/{lora_base}', n_steps)
        clear_pipes(pipe, fast_pipe)

if __name__ == '__main__':
    main()