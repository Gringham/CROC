import json
import pandas as pd


#!/usr/bin/env python

import os
import time
import argparse
import torch
from diffusers import StableDiffusion3Pipeline
from diffusers import FluxPipeline, FluxTransformer2DModel
import tqdm

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
    from para_attn.first_block_cache.diffusers_adapters import apply_cache_on_pipe
    apply_cache_on_pipe(pipe, residual_diff_threshold=0.12)
    pipe.transformer = torch.compile(pipe.transformer, mode="max-autotune-no-cudagraphs")
    print("✅ Image generation pipeline initialized successfully.")
    return pipe

def initialize_pipeline_stable_diffusion(model_id="stabilityai/stable-diffusion-3.5-large-turbo"):
    pipe = StableDiffusion3Pipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16)
    print(f"{torch.cuda.is_available()=}")
    print(f"{torch.cuda.device_count()=}")
    print(f"{torch.cuda.current_device()=}")

    pipe.transformer = torch.compile(pipe.transformer, mode="max-autotune-no-cudagraphs").to("cuda")
    print("✅ Image generation pipeline initialized successfully.")
    return pipe

def create_output_dirs(df, base_dir):
    # create one folder for every unique category_name and add "prompt" and "contrast" subfolders
    category_names = df["category_name"].unique()
    for category in category_names:
        category_dir = os.path.join(base_dir, category)
        os.makedirs(category_dir, exist_ok=True)
        os.makedirs(os.path.join(category_dir, "prompt"), exist_ok=True)
        os.makedirs(os.path.join(category_dir, "contrast"), exist_ok=True)
    
    return base_dir

def generate_images(pipe, prompt, save_dir, prompt_id, num_images=10,
                    guidance_scale=0.0, num_inference_steps=4, max_sequence_length=256):
    for img_num in range(1, num_images + 1):
        filename = f"{prompt_id}___image{img_num}.png"
        image_path = os.path.join(save_dir, filename)
        if os.path.exists(image_path):
            print(f"⏩ Skipping existing image {filename}.")
            continue
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
        print(f"✅ Saved {filename} | Time: {end - begin:.2f}s")

def multi_prompt_gen(df, model_id, out_dir="croc_hum/outputs/images", img_num=50, num_inference_steps=4, guidance_scale=0.0, max_sequence_length=256):
    # Choose the pipeline.
    if "flux" in model_id.lower(): 
        pipe = initialize_pipeline(model_id=model_id)
    else:
        pipe = initialize_pipeline_stable_diffusion(model_id=model_id)

    # Create the output directory for alt_text prompts.
    create_output_dirs(df, out_dir)
    
    prompts = df["prompt"].tolist()
    contrast_prompts = df["contrast"].tolist()
    ids = df["id"].tolist()
    category_names = df["category_name"].tolist()
    
    # Zip and iterate through the prompts and contrast prompts
    for i, (prompt, contrast_prompt, prompt_id, cat) in enumerate(zip(prompts, contrast_prompts, ids, category_names), start=1):
        generate_images(
            pipe=pipe,
            prompt=prompt,
            save_dir=os.path.join(out_dir, cat, "prompt"),
            prompt_id=prompt_id + "___" + "prompt" + "___" + model_id.split("/")[-1].replace("-", "_").replace(".", "_"),
            num_images=img_num,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            max_sequence_length=max_sequence_length
        )
        generate_images(
            pipe=pipe,
            prompt=contrast_prompt,
            save_dir=os.path.join(out_dir, cat, "contrast"),
            prompt_id=prompt_id + "___" + "contrast" + "___" + model_id.split("/")[-1].replace("-", "_").replace(".", "_"),
            num_images=img_num,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            max_sequence_length=max_sequence_length
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--type", type=str, required=True)
    parser.add_argument("--category", type=str, required=True)

    args = parser.parse_args()
    
    with open("croc_hum/outputs/core_prompts.json", "r") as file:
        data = json.load(file)

    # Prepare a list of rows for the DataFrame
    rows = []
    for key in data.keys():
        for id, details in data[key].items():
            rows.append({
                "num": id,
                "prompt": details["prompt"],
                "contrast": details["contrast"],
                "category_name": key,
                "id": key + "_" + id,
            })

    df = pd.DataFrame(rows)
    
    # Filter the DataFrame based on the category argument
    df = df[df["category_name"] == args.category]
        
    if args.type == "m1":  
        multi_prompt_gen(
            df,
            model_id="black-forest-labs/FLUX.1-dev",
            out_dir="outputs/croc_hum/images",
            img_num=50,
            num_inference_steps=50,
            guidance_scale=3.5,
            max_sequence_length=512
        )
    
    # Should be cli arg option 2
    if args.type == "m2":
        multi_prompt_gen(
            df,
            model_id="stabilityai/stable-diffusion-3.5-large",
            out_dir="outputs/croc_hum/images",
            img_num=50,
            num_inference_steps=28,
            guidance_scale=3.5,
            max_sequence_length=256
        )