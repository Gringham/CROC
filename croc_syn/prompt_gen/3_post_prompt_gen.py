import pandas as pd
import os
import json
import ast
import argparse

class PromptExplorer:
    """
    A class to load and extract prompt data from a Parquet file.
    
    Attributes:
        file_path (str): Path to the Parquet file.
        df (pd.DataFrame): DataFrame containing the prompt data.
    """
    
    def __init__(self, file_path):
        """
        Initializes the PromptExplorer with the given Parquet file.
        
        Args:
            file_path (str): Path to the Parquet file.
            
        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the file is not a Parquet file.
        """
        self.file_path = file_path
        self.df = None
        self.load_data()
    
    def load_data(self):
        """
        Loads the jsonl file into a Pandas DataFrame.
        """

        try:
            self.df = pd.read_json(self.file_path, lines=True)
        except Exception as e:
            raise Exception(f"An error occurred while loading the jsonl file: {e}")
    
    def extract_generation_outputs(self, generation_outputs_column='generated_outputs', prompts_col='prompts', contrast_prompts_col='contrast_prompts'):
        """
        Extracts 'prompt' and 'contrast_prompt' from each generation output in the 'generation_outputs' column and stores them in separate columns.
        Each entry in 'generation_outputs' is expected to be a list of 10 generation output strings.
        
        Args:
            generation_outputs_column (str): Name of the column containing the generation outputs.
            prompts_col (str): Name for the new column containing lists of extracted 'prompt's.
            contrast_prompts_col (str): Name for the new column containing lists of extracted 'contrast_prompt's.
        
        Raises:
            ValueError: If the specified generation_outputs_column does not exist.
        """
        if self.df is None:
            print("DataFrame is empty. Please load data first.\n")
            return
        
        if generation_outputs_column not in self.df.columns:
            raise ValueError(f"The column '{generation_outputs_column}' does not exist in the DataFrame.")
        

        def parse_generation_output(generation_output):
            """
            Parses a single generation output string to extract 'prompt' and 'contrast_prompt'.
            
            Args:
                generation_output (str): The generation output string containing a dictionary.
            
            Returns:
                tuple: (prompt, contrast_prompt)
            """
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
        
        for idx, row in self.df.iterrows():
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
        self.df[prompts_col] = prompts_list
        self.df[contrast_prompts_col] = contrast_prompts_list
        
        print(f"Successfully extracted '{prompts_col}' and '{contrast_prompts_col}' into separate columns.\n")
        print("Examples:", self.df[[generation_outputs_column, prompts_col, contrast_prompts_col]].head())
    
    def save_extracted_prompts(self, output_path, prompts_col='prompts', contrast_prompts_col='contrast_prompts', mode="entity_variation"):
        """
        Saves the DataFrame with extracted prompts to a new Parquet file.
        
        Args:
            output_path (str): Path to save the new Parquet file.
            prompts_col (str): Name of the column containing extracted 'prompt's.
            contrast_prompts_col (str): Name of the column containing extracted 'contrast_prompt's.
        
        Raises:
            ValueError: If the specified columns do not exist.
        """
        if self.df is None:
            print("DataFrame is empty. Please load and extract data first.\n")
            return
        
        required_columns = [prompts_col, contrast_prompts_col]
        for col in required_columns:
            if col not in self.df.columns:
                raise ValueError(f"The column '{col}' does not exist in the DataFrame. Please run 'extract_generation_outputs' first.")
        
        # Optionally, select only relevant columns to save
        #columns_to_save = [prompts_col, contrast_prompts_col]
        # If you want to keep other columns, adjust the list accordingly
        
        if mode == "entity_variation":
            self.df['property'] = None
            self.df['alt_subject'] = None
            
        columns_to_save = ['mode', 'subject', 'property', 'alt_subject', 'entity', prompts_col, contrast_prompts_col, 'model_name', "prompt_id"]
        
        # save columns to jsonl
        self.df[columns_to_save].to_json(output_path, orient='records', lines=True)
        # Print ratio of empty prompts
        total_rows = len(self.df)
        empty_prompts = self.df[prompts_col].apply(lambda x: all(not prompt.strip() for prompt in x)).sum()
        empty_contrast_prompts = self.df[contrast_prompts_col].apply(lambda x: all(not contrast_prompt.strip() for contrast_prompt in x)).sum()
        print(f"Saved extracted prompts to {output_path}.")
        print(f"Total rows: {total_rows}")
        print(f"Empty prompts: {empty_prompts} ({empty_prompts/total_rows*100:.2f}%)")
        print(f"Empty contrast prompts: {empty_contrast_prompts} ({empty_contrast_prompts/total_rows*100:.2f}%)")
            
def main():
    parser = argparse.ArgumentParser(description="Extract prompts from generated outputs and save them.")
    parser.add_argument("--mode", type=str, default="all", help="Mode for processing prompts (default: all)")
    parser.add_argument("--img_models", nargs="+", default=["SD", "FLUX"], help="List of image models to use for processing.")
    parser.add_argument("--text_models", nargs="+", default=["Qwen", "DS"], help="List of text models to use for processing.")
    parser.add_argument("--dir", type=str, default="outputs/croc_syn/prompt_gen", help="Directory to save the extracted prompts.")
    args = parser.parse_args()
    mode = args.mode
    
    in_paths = {m:{tm:os.path.join(args.dir, f"2_generated_outputs_{m.lower()}_{tm.lower()}.jsonl") for tm in args.text_models} for m in args.img_models}    
    output_paths = {m:{tm:os.path.join(args.dir, f"3_extracted_prompts_{m.lower()}_{tm.lower()}.jsonl") for tm in args.text_models} for m in args.img_models}
    
    for m in args.img_models:
        for tm in args.text_models:
            explorer = PromptExplorer(in_paths[m][tm])
    
            # Step 1: Extract 'prompt' and 'contrast_prompt' from 'generation_outputs'
            explorer.extract_generation_outputs()
            
            # Step 2: Save the extracted prompts to a new jsonl file
            explorer.save_extracted_prompts(output_paths[m][tm], mode=mode)
            
if __name__ == "__main__":
    main()
