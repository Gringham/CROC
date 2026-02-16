#!/usr/bin/env python3
import json
import os
import argparse
import tqdm
import pandas as pd
from metrics.MetricCatalogue import MetricCatalogue

def process_category(category, df, output_dir, subset, base_dir): 
    print(df.iloc[0])
    df["prompt_paths"] = df["prompt_paths"].apply(lambda x: eval(x) if isinstance(x, str) else [])
    df["contrast_paths"] = df["contrast_paths"].apply(lambda x: eval(x) if isinstance(x, str) else [])
    df["alt_contrast_paths"] = df["alt_contrast_paths"].apply(lambda x: eval(x) if isinstance(x, str) else [])
    df["prompt_paths"] = df["prompt_paths"].apply(lambda paths: [os.path.join(base_dir, p) for p in paths] if isinstance(paths, list) else [])
    df["contrast_paths"] = df["contrast_paths"].apply(lambda paths: [os.path.join(base_dir, p) for p in paths] if isinstance(paths, list) else [])
    df["alt_contrast_paths"] = df["alt_contrast_paths"].apply(lambda paths: [os.path.join(base_dir, p) for p in paths] if isinstance(paths, list) else [])
    df["prompt_paths"] = df["prompt_paths"].apply(lambda paths: [p.replace("images/", "") for p in paths] if isinstance(paths, list) else [])
    df["contrast_paths"] = df["contrast_paths"].apply(lambda paths: [p.replace("images/", "") for p in paths] if isinstance(paths, list) else [])
    df["alt_contrast_paths"] = df["alt_contrast_paths"].apply(lambda paths: [p.replace("images/", "") for p in paths] if isinstance(paths, list) else [])

    prompt_df = df[["id", "prompt", "contrast", "prompt_paths"]]
    prompt_df = prompt_df.explode("prompt_paths")


    if category == "negation":
        contrast_df = df[["id", "prompt", "contrast", "alt_contrast_paths"]].rename(columns={"alt_contrast_paths": "contrast_paths"})
    else:
        contrast_df = df[["id", "prompt", "contrast", "contrast_paths"]]
    contrast_df = contrast_df.explode("contrast_paths")

    print(f"Initial number of rows for prompt_df: {len(prompt_df)}")
    print(prompt_df.head(), contrast_df.head())
    prompt_df = prompt_df[~prompt_df["prompt_paths"].isna() & (prompt_df["prompt_paths"] != "")]
    contrast_df = contrast_df[~contrast_df["contrast_paths"].isna() & (contrast_df["contrast_paths"] != "")]
    print(f"Number of rows after filtering for prompt_df: {len(prompt_df)}")
    print(f"Number of rows after filtering for contrast_df: {len(contrast_df)}")
    


    # Set up the metric catalogue and apply the metrics
    MC = MetricCatalogue()
    MC.select_subset(subset)

    
    print(f"Processing category: {category}")
    print("Step 1: Prompt-Image pairs")
    res1 = MC.apply_subset("prompt_paths", "prompt", prompt_df, resume=True,
                suffix="___" + "orig_text" + "___" + "orig_img", 
                output_file=os.path.join(output_dir, f"{category}_prompt_img_scores.tsv"))
    
    print("Step 2: Contrast-Image pairs")
    MC.apply_subset("prompt_paths", "contrast", res1, resume=True,
                suffix="___" + "contrast_text" + "___" + "orig_img", 
                output_file=os.path.join(output_dir, f"{category}_prompt_img_scores.tsv"))
    
    print("Step 3: Contrast-Image pairs")
    res2 = MC.apply_subset("contrast_paths", "contrast", contrast_df, resume=True,
                suffix="___" + "contrast_text" + "___" + "contrast_img",
                output_file=os.path.join(output_dir, f"{category}_contrast_img_scores.tsv"))
    
    print("Step 4: Prompt-Image pairs")
    MC.apply_subset("contrast_paths", "prompt", res2, resume=True,
                suffix="___" + "orig_text" + "___" + "contrast_img",
                output_file=os.path.join(output_dir, f"{category}_contrast_img_scores.tsv"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process a specific category with the MetricCatalogue pipeline.")
    parser.add_argument("--category", type=str, default="action",
                        help="The image category to process (e.g., 'counting').")
    parser.add_argument("--ds_dir", type=str, default="croc_hum_metadata.tsv")
    parser.add_argument("--base_dir", type=str, default="croc_hum")
    parser.add_argument("--output_dir", type=str, default="outputs/croc_hum/metrics/scores")
    parser.add_argument("--subset", nargs="+", default=["CROCScore"], help="The subset of metrics to apply.")
    args = parser.parse_args()

    print("Reading metadata from:", os.path.join(args.base_dir, args.ds_dir))

    os.makedirs(args.output_dir, exist_ok=True)
    
    df = pd.read_csv(os.path.join(args.base_dir, args.ds_dir), sep="\t")
    df = df[df["id"].str.startswith(args.category)]
    process_category(args.category, df, args.output_dir, args.subset, args.base_dir)