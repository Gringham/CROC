#!/usr/bin/env python3
"""
cleanup_dataset.py

Optimized cleanup of a HuggingFace dataset without relying on `drop_duplicates`:
- Filters out examples with non-empty captions
- Ensures each example has 'subject_property' in its ID
- Also includes up to `max_entity_variations` examples whose ID contains 'entity_variation'
- Deduplicates by caption using an in-memory pass over each subset
- Uses parallel filtering for initial splits, then local dedupe (single process)
- Concatenates both subsets, shuffles, and saves
"""
import multiprocessing
from datasets import load_from_disk, Dataset, concatenate_datasets


def dedupe_by_caption(ds: Dataset) -> Dataset:
    """
    Remove duplicate captions from ds, keeping the first occurrence.
    """
    captions = ds['caption']
    seen = set()
    keep_indices = []
    for idx, cap in enumerate(captions):
        if cap not in seen:
            seen.add(cap)
            keep_indices.append(idx)
    return ds.select(keep_indices)


def main():
    # ======== Configuration ========
    input_path            = 'CROC/metrics/custom_pickscore/final_dataset'
    split                 = 'train'
    output_path           = '/metrics/custom_pickscore/clean_train'
    max_entity_variations = 3000     # up to this many 'entity_variation' IDs
    shuffle_seed          = 42       # seed for reproducible shuffling
    num_proc              = max(1, multiprocessing.cpu_count() - 1)
    # ================================

    print(f"Loading split '{split}' from {input_path}...")
    ds = load_from_disk(input_path)[split]
    print(f"Total examples: {len(ds)}")

    # 1) Remove empty captions
    print("Filtering non-empty captions...")
    ds = ds.filter(lambda ex: bool(ex.get('caption', '').strip()), num_proc=num_proc)
    print(f"After caption filter: {len(ds)}")

    # 2) Select all subject_property examples
    print("Selecting 'subject_property' examples...")
    ds_subject = ds.filter(lambda ex: 'subject_property' in ex.get('ID', ''), num_proc=num_proc)
    print(f"Found subject_property: {len(ds_subject)}")

    # 2a) Deduplicate subject_property locally
    print("Deduplicating subject_property examples by caption...")
    ds_subject = dedupe_by_caption(ds_subject)
    print(f"After dedup (subject_property): {len(ds_subject)}")

    # 3) Select entity_variation examples
    print("Selecting 'entity_variation' examples...")
    ds_entity = ds.filter(lambda ex: 'entity_variation' in ex.get('ID', ''), num_proc=num_proc)
    print(f"Found entity_variation: {len(ds_entity)}")

    # 3a) Deduplicate entity_variation locally
    print("Deduplicating entity_variation examples by caption...")
    ds_entity = dedupe_by_caption(ds_entity)
    print(f"After dedup (entity_variation): {len(ds_entity)}")

    # 4) Exclude captions already in subject_property
    print("Excluding captions already in subject_property set...")
    subject_caps = set(ds_subject['caption'])
    ds_entity = ds_entity.filter(lambda ex: ex['caption'] not in subject_caps, num_proc=num_proc)
    print(f"After excluding overlap: {len(ds_entity)}")

    # 5) Limit to max_entity_variations
    if len(ds_entity) > max_entity_variations:
        ds_entity = ds_entity.select(range(max_entity_variations))
    print(f"After limiting entity_variation to {max_entity_variations}: {len(ds_entity)}")

    # 6) Combine and shuffle
    print("Concatenating cleaned subsets...")
    cleaned: Dataset = concatenate_datasets([ds_subject, ds_entity])
    cleaned = cleaned.shuffle(seed=shuffle_seed)
    print(f"Final cleaned size: {len(cleaned)}")

    # 7) Save
    print(f"Saving cleaned dataset to {output_path}...")
    cleaned.save_to_disk(output_path)
    print("Cleanup complete.")


if __name__ == '__main__':
    main()
