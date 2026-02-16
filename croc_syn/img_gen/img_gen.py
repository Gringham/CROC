#!/usr/bin/env python

from ast import mod
import os
import time
import argparse
import pandas as pd
import torch
from diffusers import StableDiffusion3Pipeline, FluxPipeline, FluxTransformer2DModel
import tqdm
import numpy as np

def initialize_pipeline(model_id="black-forest-labs/FLUX.1-schnell"):
    transformer = FluxTransformer2DModel.from_pretrained(
        model_id,
        subfolder="transformer",
        torch_dtype=torch.bfloat16,
    )
    pipe = FluxPipeline.from_pretrained(
        model_id,
        transformer=transformer,
        torch_dtype=torch.bfloat16,
    ).to("cuda")

    # Using para attention for faster flux inference
    from para_attn.first_block_cache.diffusers_adapters import apply_cache_on_pipe
    apply_cache_on_pipe(pipe, residual_diff_threshold=0.12)
    pipe.transformer = torch.compile(pipe.transformer, mode="max-autotune-no-cudagraphs")
    return pipe

def initialize_pipeline_stable_diffusion(model_id="stabilityai/stable-diffusion-3.5-large-turbo"):
    pipe = StableDiffusion3Pipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16).to("cuda")
    pipe.transformer = torch.compile(pipe.transformer, mode="max-autotune-no-cudagraphs")
    return pipe

def generate_images(pipe, prompt, save_dir, prompt_id, id_num, num_images=10, guidance_scale=0.0, num_inference_steps=4, max_sequence_length=256):
    for img_num in range(num_images):
        filename = f"{prompt_id}_image_{img_num}.png"
        image_path = os.path.join(save_dir, filename)
        print("Trying to generate image to:", image_path)
        if os.path.exists(image_path):
            print(f"Skipping existing image {filename}.")
            continue
        try:
            begin = time.time()
            image = pipe(
                prompt,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                max_sequence_length=max_sequence_length,
                generator=torch.Generator("cpu").manual_seed(img_num)
            ).images[0]
            end = time.time()
            image.save(image_path)
            print(f"Saved {filename} | Time: {end - begin:.2f}s")
        except Exception as e:
            print(f"Failed to generate image {img_num} ID {prompt_id}, Id {id_num}: {e}")

def main(args):
    # Produce short names to use in filenames
    if "FLUX" in args.model_id:
        pipe = initialize_pipeline(model_id=args.model_id.strip())
        img_short_name = "flux"
    else:
        pipe = initialize_pipeline_stable_diffusion(model_id=args.model_id.strip())
        img_short_name = "sd"

    prompt_file_name = os.path.basename(args.input_path).split("/")[-1]
    if "_qwen" in prompt_file_name:
        text_short_name = "qwen"
    elif "_ds" in prompt_file_name:
        text_short_name = "ds"

    # Ensure the correct splitting
    assert args.num_splits % 3 == 0, "num_splits must be divisible by 3 for proper mode splitting."
    modes = ["entity_variation", "subject_property", "entity_placement"]
    mode = modes[args.split_index % len(modes)]

    current_split = args.split_index // len(modes) + 1
    total_splits = args.num_splits // len(modes)

    print(f"Processing split {current_split}/{total_splits} for mode '{mode}'.")

    # Ensure output directories exist
    prompts_dir = os.path.join(args.output_path, img_short_name, text_short_name, "prompts")
    contrast_prompts_dir = os.path.join(args.output_path, img_short_name, text_short_name, "contrast_prompts")
    os.makedirs(prompts_dir, exist_ok=True)
    os.makedirs(contrast_prompts_dir, exist_ok=True)
    
    # Read input prompts
    df = pd.read_json(args.input_path, lines=True)
    
    print("MODE:", mode)
    print("Total rows in this split before filtering by mode:", len(df))
    df = df[df["mode"] == mode]
    df = np.array_split(df, total_splits)
    print("Split df: ", df)
    df = df[current_split - 1]
    print("Total rows in this split after filtering by mode:", len(df))
    

    # Get input lists from df
    ids = df["prompt_id"].to_list()
    all_prompts = df["prompts"].to_list()
    all_contrast_prompts = df["contrast_prompts"].to_list()

    print("Commencing image generation for prompts and contrast prompts: ", len(ids), "rows to process.")
    for i, prompt_ids in tqdm.tqdm(enumerate(ids), total=len(ids)):
        prompt_list = all_prompts[i]
        contrast_prompt_list = all_contrast_prompts[i]
        if len(prompt_list) == 0:
            print(f"No prompts found for row {i}. Skipping.")
        else:
            for id_num, id_prompt in enumerate(prompt_list):
                if not id_prompt.strip():
                    print(f"Empty prompt string for row {i}, Id {id_num}. Skipping.")
                    continue
                prompt_id = prompt_ids.replace("|||", "_____") + f"_prompt_{id_num}"
                print(f"Generating images for Prompt ID {i}, Id {id_num}: '{id_prompt[:150]}...'")
                generate_images(
                    pipe=pipe,
                    prompt=id_prompt,
                    save_dir=prompts_dir,
                    prompt_id=prompt_id,
                    id_num=id_num,
                    num_images=args.num_images,
                    guidance_scale=args.guidance_scale,
                    num_inference_steps=args.num_inference_steps,
                    max_sequence_length=args.max_sequence_length
                )
        if len(contrast_prompt_list) == 0:
            print(f"No contrast prompts found for row {i}. Skipping.")
        else:
            for id_num, id_contrast_prompt in enumerate(contrast_prompt_list):
                if not id_contrast_prompt.strip():
                    print(f"Empty contrast prompt string for row {i}, Id {id_num}. Skipping.")
                    continue
                contrast_id = prompt_ids.replace("|||", "_____") + f"_contrast_{id_num}"

                print(f"Generating images for Contrast Prompt ID {i}, Id {id_num}: '{id_contrast_prompt[:150]}...'")
                generate_images(
                    pipe=pipe,
                    prompt=id_contrast_prompt,
                    save_dir=contrast_prompts_dir,
                    prompt_id=contrast_id,
                    id_num=id_num,
                    num_images=args.num_images,
                    guidance_scale=args.guidance_scale,
                    num_inference_steps=args.num_inference_steps,
                    max_sequence_length=args.max_sequence_length
                )
    print("🎉 Image generation for all prompts and contrast prompts completed successfully.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate images from prompts and contrast prompts.")
    parser.add_argument("--model_id", type=str, default="black-forest-labs/FLUX.1-schnell", help="Hugging Face model ID.")
    parser.add_argument("--input_path", type=str, default="outputs/croc_syn/prompt_gen", help="Path to the extracted prompts .jsonl file.")
    parser.add_argument("--output_path", type=str, default="outputs/croc_syn/img_gen", help="Directory where generated images will be saved.")
    parser.add_argument("--num_images", type=int, default=5, help="Number of images to generate per prompt/contrast prompt.")
    parser.add_argument("--num_inference_steps", type=int, default=4, help="Number of inference steps for the pipeline.")
    parser.add_argument("--guidance_scale", type=float, default=0.0, help="Classifier-free guidance scale.")
    parser.add_argument("--max_sequence_length", type=int, default=256, help="Maximum sequence length for text encoding in the pipeline.")
    parser.add_argument("--num_splits", type=int, default=9, help="Number of splits to divide the prompts for parallel processing. Has to be divisibly by 3")
    parser.add_argument("--split_index", type=int, default=0, help="Index of the split to process (0-based).")
    args = parser.parse_args()
    main(args)
