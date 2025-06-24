from datasets import load_from_disk, DatasetDict

# Load the dataset
dataset_path = "CROC/metrics/custom_pickscore/final_dataset_2"
dataset = load_from_disk(dataset_path)

print("Original dataset columns:")
print(dataset.column_names)

# Add "has_label" column to each split
for split in dataset:
    dataset[split] = dataset[split].add_column("num_example_per_prompt", [5] * len(dataset[split]))

# Save the modified dataset back to the same path
dataset.save_to_disk("CROC/metrics/custom_pickscore/final_dataset")
