# scripts/generate_outputs.py

import ast
import os
import pandas as pd
import argparse
from croc_syn.prompt_gen.utils.vllm_generate import vllm_generate


TEXT_MODEL_DICT = {
    "Qwen": "Qwen/QwQ-32B",
    "DS": "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
}
    

def construct_prompt_id(row: pd.Series) -> str:
    """
    Constructs a unique prompt ID based on mode, subject, alt_subject, and entity.
    
    Args:
        row (pd.Series): A row from the DataFrame.
    
    Returns:
        str: The constructed prompt ID.
    """
    mode = row['mode']
    subject = row['subject'].replace(" ", "_") if pd.notna(row['subject']) else ""
    alt_subject = row['alt_subject'].replace(" ", "_") if pd.notna(row['alt_subject']) else ""
    entity = row['entity'].replace(" ", "_") if pd.notna(row['entity']) else ""
    property = row['property'].replace(" ", "_") if pd.notna(row['property']) else ""
    
    components = [mode]
    if subject:
        components.append(subject)
    if property:
        components.append(property)
    if alt_subject:
        components.append(alt_subject)
    if entity:
        components.append(entity)
    
    prompt_id = "|||".join(components)
    return prompt_id

def extract_generation_outputs(df, generation_outputs_column='generated_outputs', prompts_col='prompts', contrast_prompts_col='contrast_prompts'): 
    def parse_generation_output(generation_output):
        if not isinstance(generation_output, str):
            return "", ""
        
        try:
            # Extract the substring from the first '{' to the last '}'
            start_index = generation_output.find("{")
            end_index = generation_output.rfind("}") + 1
            if start_index == -1 or end_index == 0:
                return "", ""
            dict_str = generation_output[start_index:end_index]
            
            # Safely evaluate the dictionary string
            parsed_dict = ast.literal_eval(dict_str)
            prompt = parsed_dict.get("prompt", "")
            contrast_prompt = parsed_dict.get("contrast_prompt", "")
            return prompt, contrast_prompt
        except (SyntaxError, ValueError, KeyError):
            return "", ""
    
    # Apply the parsing function to each generation output in the list
    prompts_list = []
    contrast_prompts_list = []
    
    for idx, row in df.iterrows():
        generation_outputs = row[generation_outputs_column]
        prompts = []
        contrast_prompts = []
        for gen_output in generation_outputs:
            prompt, contrast_prompt = parse_generation_output(gen_output)
            prompts.append(prompt)
            contrast_prompts.append(contrast_prompt)
        prompts_list.append(prompts)
        contrast_prompts_list.append(contrast_prompts)
    
    # Assign the extracted prompts to new columns
    df[prompts_col] = prompts_list
    df[contrast_prompts_col] = contrast_prompts_list
    
def main():
    arg_parser = argparse.ArgumentParser(description="Generate outputs for prompts using vLLM.")
    arg_parser.add_argument("--img_models", nargs="+", default=["FLUX", "SD"], help="List of image models to use for generation.")
    arg_parser.add_argument("--text_models", nargs="+", default=["Qwen", "DS"], help="List of text models to use for generation.")
    arg_parser.add_argument("--dir", type=str, default="outputs/croc_syn/prompt_gen", help="Directory to save the generated outputs.")
    arg_parser.add_argument("--num_generations", type=int, default=5, help="Maximum number of generations to produce for each prompt.")
    arg_parser.add_argument("--max_samples", type=str, default=None, help="Maximum number of samples to process for each image model (for testing purposes).")
    arg_parser.add_argument("--parallel", type=int, default=4, help="Number of parallel processes to use for generation.")
    args = arg_parser.parse_args()
    in_paths = {m:os.path.join(args.dir, f"1_pre_prompts_{m.lower()}.jsonl") for m in args.img_models}
    out_paths = {m:{tm:os.path.join(args.dir, f"2_generated_outputs_{m.lower()}_{tm.lower()}.jsonl") for tm in args.text_models} for m in args.img_models}    
    
    for m in args.img_models:
        print(f"Loading pre-prompts for image model '{m}' from '{in_paths[m]}'...")
        df_pre_prompts = pd.read_json(in_paths[m], lines=True)
        if args.max_samples is not None:
            df_pre_prompts = df_pre_prompts.head(int(args.max_samples))
            
        df_pre_prompts['prompt_id'] = df_pre_prompts.apply(construct_prompt_id, axis=1)
            
        all_ids = df_pre_prompts['prompt_id'].tolist()
        all_prompts = df_pre_prompts['prompt'].tolist()

        for tm in args.text_models:
            print(f"Output path for image model '{m}' and text model '{tm}': {out_paths[m][tm]}")
            
            results = vllm_generate(
                prompt_ids=all_ids,
                prompts=all_prompts,
                parallel=args.parallel,          
                num_generations=args.num_generations,
                model=TEXT_MODEL_DICT[tm],
            )
            
            df_pre_prompts['generated_outputs'] = df_pre_prompts['prompt_id'].apply(lambda pid: results.get(pid, []))
            
            extract_generation_outputs(df_pre_prompts)
            
            print(f"Saving generated outputs to '{out_paths[m][tm]}'...")
            df_pre_prompts.to_json(out_paths[m][tm], orient='records', lines=True)
                
if __name__ == "__main__":
    main()
