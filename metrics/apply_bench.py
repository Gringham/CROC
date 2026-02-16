from ast import parse
from calendar import c
import argparse, os
import numpy as np
import pandas as pd
from regex import P
import numpy as np

from metrics.MetricCatalogue import MetricCatalogue

def generate_image_paths(img_dir, row, category, folder, n=0, img_num=5, prompt_num=10):
    return [[f"{img_dir}/{folder}/{row['prompt_id'].replace('|||', '_____')}_{category}_{pn+n}_image_{inum+n}.png" for inum in range(img_num)] for pn in range(prompt_num)]

def apply_bench(df, part, output_dir="outputs/croc_syn/metrics/scores"):

    prompt_df = pd.DataFrame({
        "prompt": df["prompts"].tolist(),
        "contrast_prompt": df["contrast_prompts"].tolist(),
        "img_paths_prompts": df["img_paths_prompts"].tolist(),
        "img_paths_contrast": df["img_paths_contrast"].tolist(),
        "prompt_id": df["prompt_id"],
        
    })

    MC = MetricCatalogue()
    MC.select_subset(["BVQA", "CLIPScore_Large", "SSAlign", "PickScore", "VQAScore", "Blip2ITM"])
    
    res1 = MC.apply_subset("img_paths_prompts", "prompt", prompt_df, resume=True,
                suffix="___" + "orig_text" + "___" + "orig_img", 
                output_file=os.path.join(output_dir, f"part{part}_scores.tsv"))
    
    res2 = MC.apply_subset("img_paths_prompts", "contrast_prompt", res1, resume=True,
                suffix="___" + "contrast_text" + "___" + "orig_img", 
                output_file=os.path.join(output_dir, f"part{part}_scores.tsv"))
    
    res3 = MC.apply_subset("img_paths_contrast", "contrast_prompt", res2, resume=True,
                suffix="___" + "contrast_text" + "___" + "contrast_img",
                output_file=os.path.join(output_dir, f"part{part}_scores.tsv"))
    
    MC.apply_subset("img_paths_contrast", "prompt", res3, resume=True,
                suffix="___" + "orig_text" + "___" + "contrast_img",
                output_file=os.path.join(output_dir, f"part{part}_scores.tsv"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process a specific category with the MetricCatalogue pipeline.")
    parser.add_argument("--prompt_file", type=str, default="outputs/croc_syn/prompt_gen/3_extracted_prompts_flux_ds.jsonl", help="Path to the prompt file.")
    parser.add_argument("--img_dir", type=str, default="outputs/croc_syn/img_gen/flux/qwen", help="Directory containing the images.")
    parser.add_argument("--chunk_part", type=int, default=0, help="Which chunk to process.") 
    parser.add_argument("--chunk_num", type=int, default=10, help="Total number of chunks to split the data into.")
    parser.add_argument("--img_num", type=int, default=5, help="Number of images per prompt.")
    parser.add_argument("--prompt_num", type=int, default=10, help="Number of prompts per row.")
    parser.add_argument("--base_idx", type=int, default=0, help="Base index for image numbering.")
    parser.add_argument("--filter_mode", type=str, default="all", help="Filter mode for prompts.")
    parser.add_argument("--output_dir", type=str, default="outputs/croc_syn/metrics/scores", help="Directory to save the output scores.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    
    print("Loading prompts from file: ", args.prompt_file)
    df = pd.read_json(args.prompt_file, lines=True)
    if args.filter_mode != "all": 
        df = df[df["mode"] == args.filter_mode]
        
    df = np.array_split(df, args.chunk_num)
    print("Split df: ", df)
    df = df[args.chunk_part]
    print("Selected chunk: ", df.iloc[0])
    
        
    df['img_paths_prompts'] = df.apply(lambda x: generate_image_paths(args.img_dir, x, "prompt", "prompts", img_num=args.img_num, prompt_num=args.prompt_num, n=args.base_idx), axis=1)
    df['img_paths_contrast'] = df.apply(lambda x: generate_image_paths(args.img_dir, x, "contrast", "contrast_prompts", img_num=args.img_num, prompt_num=args.prompt_num, n=args.base_idx), axis=1)
    
    df = df.explode(["prompts", "contrast_prompts", "img_paths_prompts", "img_paths_contrast"], ignore_index=True)
    df = df.explode(["img_paths_prompts", "img_paths_contrast"], ignore_index=True)

    df = df[df["prompts"].str.strip() != ""]
    df = df[df["contrast_prompts"].str.strip() != ""]

    print(df.iloc[0])

    if len(df) == 0:
        print("No prompts found for filter mode: ", args.filter_mode)
        exit()

    # Drop non existent image paths
    print("Sample Paths: ", df["img_paths_prompts"][:5].to_list())
    print("Sample Contrast Paths: ", df["img_paths_contrast"][:5].to_list())
    mask_prompts = df['img_paths_prompts'].apply(os.path.exists)
    mask_constrast = df['img_paths_contrast'].apply(os.path.exists)
    df_missing = df[~(mask_prompts & mask_constrast)]
    df = df[mask_prompts & mask_constrast].reset_index(drop=True)
    dropped_rows = len(mask_prompts) - mask_prompts.sum()
    if dropped_rows > 0:
        print(f"Warning: Dropped {dropped_rows} rows due to missing images. E.g.: {df_missing['img_paths_prompts'].iloc[0]} and {df_missing['img_paths_contrast'].iloc[0]}")

    df["prompt_id"] = df["prompt_id"]  + "_____" + [str(i+1) for i in range(args.img_num)] * int((len(df)/args.img_num))

    apply_bench(df, args.chunk_part, output_dir=args.output_dir)