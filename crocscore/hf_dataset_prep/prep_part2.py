from datasets import load_from_disk, Dataset, DatasetDict, Features, Value
import random

# -------------------------------
# Step 1: Load the Hugging Face Dataset from Disk
# -------------------------------
dataset_path = "CROC/metrics/custom_pickscore/local_dataset"
dataset = load_from_disk(dataset_path)
print("Original dataset columns:")
print(dataset.column_names)

# -------------------------------
# Step 2: Define a transformation function that creates consistent output rows
# -------------------------------
def duplicate_and_augment(example):
    """
    Transforms an example into two new rows.
    Each output row will have the same keys: "ID", "caption", "label_0", and "label_1".
    
    - Row 1: Uses the value from 'prompts' as the caption, and sets label_0=1, label_1=0.
    - Row 2: Uses the value from 'contrast_prompts' as the caption, and sets label_0=0, label_1=1.
    """
    row1 = {
        "ID": example.get("ID"),
        "caption": example.get("prompts", ""),
        "label_0": 1,
        "label_1": 0,
        "are_different": True,
        "jpg_0": example.get("img_jpeg_base64_prompts", ""),
        "jpg_1": example.get("img_jpeg_base64_contrast", ""),
        "model_0": example.get("img_gen_model", ""),
        "model_1": example.get("img_gen_model", ""),
        "text_model_0": example.get("text_gen_model", ""),
        "text_model_1": example.get("text_gen_model", ""),
        "image_0_url": example.get("img_paths_prompts", ""),
        "image_1_url": example.get("img_paths_contrast", ""),
    }
    row2 = {
        "ID": example.get("ID"),
        "caption": example.get("contrast_prompts", ""),
        "label_0": 0,
        "label_1": 1,
        "are_different": True,
        "jpg_0": example.get("img_jpeg_base64_prompts", ""),
        "jpg_1": example.get("img_jpeg_base64_contrast", ""),
        "model_0": example.get("img_gen_model", ""),
        "model_1": example.get("img_gen_model", ""),
        "text_model_0": example.get("text_gen_model", ""),
        "text_model_1": example.get("text_gen_model", ""),
        "image_0_url": example.get("img_paths_prompts", ""),
        "image_1_url": example.get("img_paths_contrast", ""),
        
        
    }
    return [row1, row2]

# -------------------------------
# Step 3: Create a lazy (generator-based) augmented dataset
# -------------------------------
def transform_generator():
    # Iterate over the original dataset lazily.
    for example in dataset:
        # For each example, yield both augmented rows.
        for new_example in duplicate_and_augment(example):
            yield new_example

# Define explicit features for the new dataset to enforce consistent types.
features = Features({
    "ID": Value("string"),
    "caption": Value("string"),
    "label_0": Value("int32"),
    "label_1": Value("int32"),
    "are_different": Value("bool"),
    "jpg_0": Value("string"),
    "jpg_1": Value("string"),
    "model_0": Value("string"),
    "model_1": Value("string"),
    "text_model_0": Value("string"),
    "text_model_1": Value("string"),
    "image_0_url": Value("string"),
    "image_1_url": Value("string"),
})

# Create the augmented dataset from the generator with the explicit schema.
augmented_dataset = Dataset.from_generator(transform_generator, features=features)
print("Transformed dataset columns:")
print(augmented_dataset.column_names)

# -------------------------------
# Step 4: Split the augmented dataset into train, dev, and test sets based on unique "ID" values
# -------------------------------
# Get the list of unique IDs (each duplicate pair still shares the same "ID").
unique_ids = list(set(augmented_dataset["ID"]))
print(f"Total unique IDs: {len(unique_ids)}")

# Shuffle the unique IDs to randomize the splits (fixed seed for reproducibility).
random.seed(42)
random.shuffle(unique_ids)

# Define split sizes: 80% train, 10% dev, 10% test.
n_ids = len(unique_ids)
n_train = int(0.8 * n_ids)
n_dev = int(0.1 * n_ids)

train_ids = set(unique_ids[:n_train])
dev_ids   = set(unique_ids[n_train:n_train+n_dev])
test_ids  = set(unique_ids[n_train+n_dev:])

# Use the filter method with progress descriptions (the filtering is performed lazily).
train_dataset = augmented_dataset.filter(lambda x: x["ID"] in train_ids, desc="Filtering train")
dev_dataset   = augmented_dataset.filter(lambda x: x["ID"] in dev_ids,   desc="Filtering dev")
test_dataset  = augmented_dataset.filter(lambda x: x["ID"] in test_ids,  desc="Filtering test")

# Combine the splits into a DatasetDict.
final_dataset = DatasetDict({
    "train": train_dataset,
    "dev": dev_dataset,
    "test": test_dataset
})

print("Final Dataset splits:")
for split, ds in final_dataset.items():
    print(f"{split}: {ds.num_rows} rows")

# -------------------------------
# Step 5: Save the final dataset to disk
# -------------------------------
final_save_path = "CROC/metrics/custom_pickscore/final_dataset"
final_dataset.save_to_disk(final_save_path)
print(f"Final dataset saved to: {final_save_path}")
