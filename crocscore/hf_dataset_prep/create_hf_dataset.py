import os
import pandas as pd
from datasets import Dataset
from PIL import Image
import io
import base64
from tqdm import tqdm
import concurrent.futures

# Define your path dictionaries.
path1 = {
    "path": "CROC/datasets/post_prompt_gen_transformed/qwen_qwq_32b/all_prompts_all_cot_with_generated_outputs_sd_new.parquet",
    "text_gen_model": "qwen_qwq_32b",
    "image_gen_model": "stable-diffusion",
    "image_dir": "CROC/outputs/images/qwen_qwq_32b/stable_diffusion_3_5_large_turbo",
}
path2 = {
    "path": "CROC/datasets/post_prompt_gen_transformed/deepseek_r1_distill_qwen_14b/all_prompts_all_cot_with_generated_outputs_sd_new.parquet",
    "text_gen_model": "deepseek_r1_distill_qwen_14b",
    "image_gen_model": "stable-diffusion",
    "image_dir": "CROC/outputs/images/deepseek_r1_distill_qwen_14/stable_diffusion_3_5_large_turbo",
}
path3 = {
    "path": "CROC/datasets/post_prompt_gen_transformed/qwen_qwq_32b/extracted_prompts_all_cot.parquet",
    "text_gen_model": "qwen_qwq_32b",
    "image_gen_model": "flux",
    "image_dir": "CROC/outputs/images/qwen_qwq_32b/flux1_schnell",
}
path4 = {
    "path": "CROC/datasets/post_prompt_gen_transformed/deepseek_r1_distill_qwen_14b/extracted_prompts_all_cot.parquet",
    "text_gen_model": "deepseek_r1_distill_qwen_14b",
    "image_gen_model": "flux",
    "image_dir": "CROC/outputs/images/deepseek_r1_distill_qwen_14/flux1_schnell",
}

# Read each parquet file, load the DataFrame, and add model metadata.
paths = [path1, path2, path3, path4]
frames = []
for path in paths:
    df_tmp = pd.read_parquet(path["path"])
    df_tmp["text_gen_model"] = path["text_gen_model"]
    df_tmp["image_gen_model"] = path["image_gen_model"]
    # (Optionally add the image directory; this can be useful later.)
    df_tmp["image_dir"] = path["image_dir"]
    
    if "prompt" in df_tmp.columns:
        df_tmp.drop(columns=["prompt"], inplace=True)
    if "generated_outputs" in df_tmp.columns:
        df_tmp.drop(columns=["generated_outputs"], inplace=True)
    
    # Assume the number of prompts is given by the length of the list in the first row of the "prompts" column.
    prompt_num = len(df_tmp["prompts"].iloc[0])
    
    # Create an ID based on several fields (replace spaces with underscores and ignore None values).
    df_tmp["ID"] = df_tmp.apply(
        lambda x: "_".join([
            x["mode"] if x["mode"] and x["mode"] != "None" else "",
            x["subject"].replace(" ", "_") if x["subject"] and x["subject"] != "None" else "",
            x["property"].replace(" ", "_") if x["property"] and x["property"] != "None" else "",
            x["alt_subject"].replace(" ", "_") if x["alt_subject"] and x["alt_subject"] != "None" else "",
            x["entity"].replace(" ", "_") if x["entity"] else ""
        ]).strip("_"),
        axis=1
    )
    
    # Function to generate image paths.
    def generate_image_paths(row, category, folder, image_dir):
        # Build a nested list: one list for each prompt, each containing 8 image paths.
        return [
            [f"{image_dir}/{folder}/{category}{pn+1}_{row['ID']}_{category}{pn+1}_image{inum+1}.png" 
             for inum in range(8)]
            for pn in range(prompt_num)
        ]
    
    # Generate image paths for prompts and contrast images.
    df_tmp['img_paths_prompts'] = df_tmp.apply(lambda x: generate_image_paths(x, "prompt", "prompts", x["image_dir"]), axis=1)
    df_tmp['img_paths_contrast'] = df_tmp.apply(lambda x: generate_image_paths(x, "contrast", "contrast_prompts", x["image_dir"]), axis=1)
    
    frames.append(df_tmp)

# Concatenate the frames vertically.
df = pd.concat(frames, ignore_index=True)

# Explode the nested list columns so that each row represents a single image path.
# (Explode the textual columns first if needed.)
df = df.explode(["prompts", "contrast_prompts", "img_paths_prompts", "img_paths_contrast"], ignore_index=True)
df = df.explode(["img_paths_prompts", "img_paths_contrast"], ignore_index=True)

# Keep only rows where both image paths exist.
mask_prompts = df['img_paths_prompts'].apply(os.path.exists)
mask_contrast = df['img_paths_contrast'].apply(os.path.exists)
df = df[mask_prompts & mask_contrast].reset_index(drop=True)


# ==================================================
# New section: convert PNG images to JPEG and encode in base64.
# ==================================================

def process_image(image_path):
    """
    Open a PNG image from the given path, convert to JPEG (RGB),
    and return a base64-encoded string of the JPEG bytes.
    """
    try:
        with Image.open(image_path) as img:
            img = img.convert("RGB")  # Ensure no alpha channel
            buffer = io.BytesIO()
            img.save(buffer, format="JPEG")
            byte_data = buffer.getvalue()
            encoded_str = base64.b64encode(byte_data).decode("utf-8")
            return encoded_str
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None

# Process images in the 'img_paths_prompts' column.
prompts_paths = df["img_paths_prompts"].tolist()
with concurrent.futures.ProcessPoolExecutor() as executor:
    base64_prompts = list(tqdm(executor.map(process_image, prompts_paths), 
                               total=len(prompts_paths), desc="Processing prompts images"))
df["img_jpeg_base64_prompts"] = base64_prompts

# Process images in the 'img_paths_contrast' column.
contrast_paths = df["img_paths_contrast"].tolist()
with concurrent.futures.ProcessPoolExecutor() as executor:
    base64_contrast = list(tqdm(executor.map(process_image, contrast_paths), 
                                total=len(contrast_paths), desc="Processing contrast images"))
df["img_jpeg_base64_contrast"] = base64_contrast

# ==================================================
# Create a Hugging Face Dataset from the DataFrame.
hf_dataset = Dataset.from_pandas(df)

# Optionally, inspect the dataset structure.
print(hf_dataset)
print(hf_dataset.column_names)

# Save the final dataset to disk.
hf_dataset.save_to_disk("CROC/metrics/custom_pickscore/local_dataset")
