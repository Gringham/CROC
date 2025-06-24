from ast import parse
from calendar import c
import argparse, os
import numpy as np
import pandas as pd
from regex import P

from vmen.metrics.MetricCatalogue import MetricCatalogue

def apply_own_bench(prompt1="orig_text", 
                    img1="orig_img", 
                    prompt2="contrast_text", 
                    img2="orig_img", 
                    prompt_file="CROC/datasets/post_prompt_gen_transformed/deepseek_r1_distill_qwen_14B/extracted_prompts_all_cot.parquet",
                    dsg_question_cache=None,
                    dsg_dependency_cache=None,
                    image_dir="CROC/outputs",
                    resume=True,
                    output_files=None,
                    subset=None,
                    filter_mode="all",
                    img_num=10,
                    eval_mode="1_1",
                    chunk_num=None,
                    use_alt_prev=False):


    df = pd.read_parquet(prompt_file)
    if filter_mode != "all": 
        df = df[df["mode"] == filter_mode]
    
    print(df)
    df["ID"] = df.apply(
        lambda x: "_".join([
            x["mode"] if x["mode"] and x["mode"] != "None" else "",
            x["subject"].replace(" ", "_") if x["subject"] and x["subject"] != "None" else "",
            x["property"].replace(" ", "_") if x["property"] and x["property"] != "None" else "",
            x["alt_subject"].replace(" ", "_") if x["alt_subject"] and x["alt_subject"] != "None" else "",
            x["entity"].replace(" ", "_") if x["entity"] else ""
        ]).strip("_"), 
        axis=1
    )
        
    prompt_num = len(df["prompts"].iloc[0]) 
        
    def generate_image_paths(row, category, folder):
        return [[f"{image_dir}/{folder}/{category}{pn+1}_{row['ID']}_{category}{pn+1}_image{inum+1}.png" for inum in range(img_num)] for pn in range(prompt_num)]
            
    df['img_paths_prompts'] = df.apply(lambda x: generate_image_paths(x, "prompt", "prompts"), axis=1)
    df['img_paths_contrast'] = df.apply(lambda x: generate_image_paths(x, "contrast", "contrast_prompts"), axis=1)
    
    df = df.explode(["prompts", "contrast_prompts", "img_paths_prompts", "img_paths_contrast"], ignore_index=True)
    df = df.explode(["img_paths_prompts", "img_paths_contrast"], ignore_index=True)
        
    if prompt1 == "orig_text":
        text_prompt1 = "prompts"
    elif prompt1 == "contrast_text":
        text_prompt1 = "contrast_prompts"
    if img1 == "orig_img":
        path_img_1 = "img_paths_prompts"
    elif img1 == "contrast_img":
        path_img_1 = "img_paths_contrast"
    if prompt2 == "orig_text":
        text_prompt2 = "prompts"
    elif prompt2 == "contrast_text":
        text_prompt2 = "contrast_prompts"
    if img2 == "orig_img":
        path_img_2 = "img_paths_prompts"
    elif img2 == "contrast_img":
        path_img_2 = "img_paths_contrast"
        
    # Drop non existent image paths
    print("Sample Paths: ", df["img_paths_prompts"][:5].to_list())
    mask_prompts = df['img_paths_prompts'].apply(os.path.exists)
    mask_constrast = df['img_paths_contrast'].apply(os.path.exists)
    df = df[mask_prompts & mask_constrast].reset_index(drop=True)

    print(f"Number of prompts: {len(df)}")
    print("First 3 rows: ", df.head(3))
    
    df.to_csv("test.csv")


    MC = MetricCatalogue(dsg_question_cache=dsg_question_cache, dsg_dependency_cache=dsg_dependency_cache)
    MC.select_subset(subset)
    
    # Drop the "generated_outputs" column if it exists. Otherwise the files will be too large.
    if "generated_outputs" in df.columns:
        df = df.drop(columns=["generated_outputs"])
    if "prompt" in df.columns:
        df = df.drop(columns=["prompt"])
        
        
    if use_alt_prev:
        alt_prev = os.path.join(output_files, f"{filter_mode}___{'_'.join(subset)}___{eval_mode}.tsv")
    else:
        alt_prev = None
        
    # Split the dataframe into 8 chunks and select the chunk_num-th chunk
    if chunk_num is not None:
        chunk = np.array_split(df, 8)[chunk_num]
    else:
        chunk = df
    
    chunk = MC.apply_subset(path_img_1, text_prompt1, chunk, resume=resume, suffix="___" + prompt1 + "___" + img1, output_file = os.path.join(output_files, f"{filter_mode}___{'_'.join(subset)}___{eval_mode}___{chunk_num}.tsv"),
                         alt_prev_file = alt_prev)
    df = MC.apply_subset(path_img_2, text_prompt2, chunk, resume=resume, suffix="___" + prompt2 + "___" + img2, output_file = os.path.join(output_files, f"{filter_mode}___{'_'.join(subset)}___{eval_mode}___{chunk_num}.tsv"))

    return df

def main():
    print("Running Benchmarking Script")
    parser = argparse.ArgumentParser(description="Run t2i_bench with different configurations.")
    
    parser.add_argument(
        '--setup', 
        type=str, 
        choices=['1_1', '1_1_inverse', "2_1", "2_1_inverse"],   
        default='1_1',
        help='Choose which setup to run.'
    )
    
    parser.add_argument(
    '--resume',
    action='store_true',
    help='Resume from backup file.'
    )
    
    parser.add_argument(
    '--prompt_file',
    type=str,  # Change from 'action=store_true' to 'type=str'
    default="CROC/datasets/post_prompt_gen_transformed/deepseek_r1_distill_qwen_14b/extracted_prompts_all_cot.parquet",
    help='Input prompts file'
    )

    parser.add_argument(
        '--img_dir',
        type=str,  # Change from 'action=store_true' to 'type=str'
        default="CROC/outputs/images/deepseek_r1_distill_qwen_14b/stable_diffusion_3_5_large_turbo",
        help='Generated images directory'
    )

    parser.add_argument(
        '--dsg_question_cache',
        type=str,  # Change from 'action=store_true' to 'type=str'
        default='CROC/backup/dsgq1.pkl',
        help='DSG question cache file'
    )

    parser.add_argument(
        '--dsg_dependency_cache',
        type=str,  # Change from 'action=store_true' to 'type=str'
        default='CROC/backup/dsgd1.pkl',
        help='DSG dependency cache file'
    )

    parser.add_argument(
        '--output_files',
        type=str,  # Change from 'action=store_true' to 'type=str'
        default='CROC/outputs/metric_results_deepseek_sd/own_bench',
        help='Output directory for results'
    )

    parser.add_argument(
        '--subset',
        type=lambda s: s.split(','),  # Parse comma-separated values
        default=["SSAlign"],
        help='Subset of metrics to apply (comma-separated)'
    )

    parser.add_argument(
        '--filter_mode',
        type=str,  # Ensure this accepts a string
        default="entity_placement",
        help='Filter mode'
    )
    
    parser.add_argument(
        '--chunk_num',
        type=int,
        default=None,
        help='Chunk number to process'
    )

    parser.add_argument(
        '--use_alt_prev',
        action='store_true',
        help='Use alternative previous file'
    )

    args = parser.parse_args()
    
    args.resume = True
    
    print("Running with the following arguments:")
    print(f"Setup: {args.setup}")
    print(f"Resume: {args.resume}")
    print(f"Prompt file: {args.prompt_file}")
    print(f"Image directory: {args.img_dir}")
    print(f"DSG question cache: {args.dsg_question_cache}")
    print(f"DSG dependency cache: {args.dsg_dependency_cache}")
    print(f"Output files: {args.output_files}")
    print(f"Subset: {args.subset}")
    print(f"Filter mode: {args.filter_mode}")
    print(f"Chunk number: {args.chunk_num}")
    
    if args.setup == '1_1':
        apply_own_bench(
            prompt1="orig_text",
            img1="orig_img",
            prompt2="contrast_text",
            img2="orig_img",
            prompt_file=args.prompt_file,
            dsg_question_cache=args.dsg_question_cache,
            dsg_dependency_cache=args.dsg_dependency_cache,
            image_dir=args.img_dir,
            resume=args.resume,
            output_files=args.output_files,
            subset=args.subset,
            filter_mode=args.filter_mode,
            eval_mode="1_1",
            chunk_num=args.chunk_num,
            use_alt_prev=args.use_alt_prev
        )

    elif args.setup == '1_1_inverse':
        apply_own_bench(
            prompt1="contrast_text",
            img1="contrast_img",
            prompt2="orig_text",
            img2="contrast_img",
            prompt_file=args.prompt_file,
            dsg_question_cache=args.dsg_question_cache,
            dsg_dependency_cache=args.dsg_dependency_cache,
            image_dir=args.img_dir,
            resume=args.resume,
            output_files=args.output_files,
            subset=args.subset,
            filter_mode=args.filter_mode,
            eval_mode="1_1_inverse",
            chunk_num=args.chunk_num,
            use_alt_prev=args.use_alt_prev
        )
    elif args.setup == '2_1':
        apply_own_bench(
            prompt1="orig_text",
            img1="orig_img",
            prompt2="orig_text",
            img2="contrast_img",
            prompt_file=args.prompt_file,
            dsg_question_cache=args.dsg_question_cache,
            dsg_dependency_cache=args.dsg_dependency_cache,
            image_dir=args.img_dir,
            resume=args.resume,
            output_files=args.output_files,
            subset=args.subset,
            filter_mode=args.filter_mode,
            eval_mode="2_1",
            chunk_num=args.chunk_num,
            use_alt_prev=args.use_alt_prev
        )

    elif args.setup == '2_1_inverse':
        apply_own_bench(
            prompt1="contrast_text",
            img1="contrast_img",
            prompt2="contrast_text",
            img2="orig_img",
            prompt_file=args.prompt_file,
            dsg_question_cache=args.dsg_question_cache,
            dsg_dependency_cache=args.dsg_dependency_cache,
            image_dir=args.img_dir,
            resume=args.resume,
            output_files=args.output_files,
            subset=args.subset,
            filter_mode=args.filter_mode,
            eval_mode="2_1_inverse",
            chunk_num=args.chunk_num,
            use_alt_prev=args.use_alt_prev
        )


if __name__ == "__main__":
    main()