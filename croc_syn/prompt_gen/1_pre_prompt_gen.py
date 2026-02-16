import json
import os
import random
from itertools import product, islice

import numpy as np
import pandas as pd
import argparse

TEMPLATES = {
    "subject_property": """Consider the following guide on writing a good prompt with {model_name}:
{guide}
Write a prompt for {model_name} that describes a specific scene about the subject {subject_name} ({subject_description}) that also involves the concept "{property_name}" ({property_description}).
Additionally, write a contrast prompt that strongly contrasts the original prompt in terms of the concept "{property_name}" ({property_description}), but keeps the wording and content of the prompt the same as far as possible.
For example, if the concept is a color the contrast prompt may use a different color.
Pay attention not to use unusual words and make sure that the contents can be displayed as images. Use specific and understandable. Do not use phrases like "the same" in the contrast prompt.
 Think step by step, then return your output in the following format:
{{
    "prompt": "Your prompt here",
    "contrast_prompt": "Your contrast prompt here"
}}
""",

    "entity_variation": """Consider the following guide on writing a good prompt with {model_name}:
{guide}
Write a prompt for {model_name} that describes a specific scene about the subject {subject_name} ({subject_description}) involving the entity {entity_name} (Definition: {entity_description}).
Additionally, write a contrast prompt that strongly changes parts of the entity definition {entity_name} (Definition: {entity_description}), but keeps the wording and content of the prompt the same as far as possible.
For example, if the entity is a human that has two arms, the contrast prompt may change the number of arms to three.
Pay attention not to use unusual words and make sure that the contents can be displayed as images. Use specific and understandable language. Do not use phrases like "the same" in the contrast prompt.
Think step by step, then return your output in the following format:
{{
    "prompt": "Your prompt here",
    "varied_definition": "Strongly changed definition of {entity_name} (Definition: {entity_description}. The definition needs to be displayable as an image and it should change the visual appearance of the entity in an unexpected way, ideally not by adding external elements, for example by changing the shape, color or changing numbers.)"
    "contrast_prompt": "Your contrast prompt here"
}}
""",

    "entity_placement": """Consider the following guide on writing a good prompt with {model_name}:
{guide}
Write a prompt for {model_name} that describes a specific scene about the subject {subject_name} ({subject_description}) with the entity {entity_name} (Definition: {entity_description}).
Additionally, write a contrast prompt that places the entity {entity_name} in a picture about the subject {alt_subject_name} ({alt_subject_description}), but keeps the wording and content of the prompt the same as far as possible.
Pay attention not to use unusual words and make sure that the contents can be displayed as images. Use specific and understandable language. Do not use phrases like "the same" in the contrast prompt.
Think step by step, then return your output in the following format:
{{
    "prompt": "Your prompt here",
    "contrast_prompt": "Your contrast prompt here"
}}
"""
}

GUIDE_DICT = {
    "FLUX": "croc_syn/prompt_gen/utils/flux_guide.txt",
    "SD": "croc_syn/prompt_gen/utils/sd_guide.txt"
}




def find_class(tax, target):
    """
    Recursively searches 'tax' for a class/node with name == target.
    Returns that node if found, else None.
    """
    if tax.get('name') == target:
        return tax
    for c in tax.get('classes', []):
        result = find_class(c, target)
        if result:
            return result
    return None


def get_leaves(tax, previous_nodes=None):
    """
    Returns a list of { 'leaf': node, 'path': [...hierarchy...] } for every leaf node 
    (i.e., node that doesn't have sub-classes).
    """
    if previous_nodes is None:
        previous_nodes = []
    if 'classes' not in tax or not tax['classes']:
        return [{'leaf': tax, 'path': previous_nodes}]
    leaves = []
    for c in tax.get('classes', []):
        leaves.extend(get_leaves(c, previous_nodes + [tax]))
    return leaves


def get_combinations(taxonomy, mode="subject_property"):
    """
    Loads the taxonomy and returns a list of combinations to feed into build_pgp_prompts.
    
    The structure depends on 'mode':
      - subject_property: pairs of (subject leaf, other leaf).
      - entity_variation: pairs of (subject leaf, entity).
      - entity_placement: 
          For the core subject, choose multiple alt subjects.
          Do not iterate all combinations exhaustively; limit to 100 samples.
    """
    # ------------------------------------
    # subject_property
    # ------------------------------------
    if mode == "subject_property":
        # 1. Find "Subject Matter" node
        subject_node = find_class(taxonomy, "Subject Matter") # Called topics in the paper
        if not subject_node:
            raise ValueError("Could not find a 'Subject Matter' node in the taxonomy.")

        # 2. Get all subject leaves
        subject_leaves = get_leaves(subject_node)

        # 3. Get all leaves in the entire taxonomy (for the 'property' side)
        all_leaves = get_leaves(taxonomy)

        # 4. Exclude subject leaves to get the 'other leaves'
        other_leaves = [
            leaf for leaf in all_leaves
            if all(leaf['leaf'] != subj_leaf['leaf'] for subj_leaf in subject_leaves)
        ]

        # 5. Build cross product
        combinations = []
        for sm, oc in product(subject_leaves, other_leaves):
            combinations.append({
                'subject': sm,
                'property': oc
            })

    # ------------------------------------
    # entity_variation
    # ------------------------------------
    elif mode == "entity_variation":
        # 1. Find "Subject Matter" node to get subjects
        subject_node = find_class(taxonomy, "Subject Matter")
        if not subject_node:
            raise ValueError("Could not find a 'Subject Matter' node in the taxonomy.")

        # 2. Get all subject leaves
        subject_leaves = get_leaves(subject_node)

        # 3. Get all leaves in the entire taxonomy
        all_leaves = get_leaves(taxonomy)

        # 4. Gather all entities from any leaf that has an 'entities' field
        entities = []
        for leaf_dict in all_leaves:
            node = leaf_dict['leaf']
            if 'entities' in node and isinstance(node['entities'], list):
                for entity in node['entities']:
                    entities.append(entity)

        # 5. Define parent subject
        parent_subject = {'leaf': subject_node, 'path': []}

        # 6. For reproducibility
        random.seed(42)

        # 7. Create limited subject-entity pairs
        combinations = []
        for entity in entities:
            # Select 9 random subjects from subject_leaves
            if len(subject_leaves) >= 9:
                selected_subjects = random.sample(subject_leaves, 9)
            else:
                selected_subjects = subject_leaves.copy()

            # Add parent_subject to make it 10 subjects per entity
            selected_subjects.append(parent_subject)

            # Create subject-entity pairs
            for subject in selected_subjects:
                combinations.append({
                    'subject': subject,
                    'entity': entity
                })

    # ------------------------------------
    # entity_placement
    # ------------------------------------
    elif mode == "entity_placement":
        # 1. We gather all leaves
        all_leaves = get_leaves(taxonomy)

        # 2. We also need the subject leaves from "Subject Matter" to pick an alt subject
        subject_node = find_class(taxonomy, "Subject Matter")
        if not subject_node:
            raise ValueError("Could not find a 'Subject Matter' node in the taxonomy.")
        subject_leaves = get_leaves(subject_node)

        # For reproducibility:
        random.seed(42)

        combinations = []
        for leaf_dict in all_leaves:
            node = leaf_dict['leaf']

            # If this leaf has an 'entities' field, we treat this leaf as the "core subject"
            if 'entities' in node and isinstance(node['entities'], list):
                # Determine how many samples to generate per entity
                num_samples = 10  # Increased number of samples

                # Prepare list of possible alt_subjects excluding the current subject
                alt_subject_candidates = [
                    subj_leaf for subj_leaf in subject_leaves 
                    if subj_leaf['leaf'] != leaf_dict['leaf']  # exclude the same leaf
                ]

                if not alt_subject_candidates:
                    continue

                # If there are fewer candidates than num_samples, adjust accordingly
                actual_samples = min(num_samples, len(alt_subject_candidates))

                # Randomly sample alt_subjects without replacement
                sampled_alt_subjects = random.sample(alt_subject_candidates, actual_samples)

                for entity in node['entities']:
                    for alt_subject in sampled_alt_subjects:
                        combinations.append({
                            "subject": leaf_dict,       # the leaf that "owns" the entity
                            "alt_subject": alt_subject, # a sampled alt subject
                            "entity": entity
                        })
                        

    else:
        raise ValueError(f"Unknown mode: {mode}")

    return combinations


def build_pgp_prompts(guide, combinations, model_name="FLUX 1.", mode="subject_property", templates=TEMPLATES):
    """
    Creates a list of prompts from 'combinations', using the template that corresponds
    to 'mode' in the given 'templates' dictionary.
    """
    # Grab the template string for the given mode
    prompt_template = templates.get(mode)
    if not prompt_template:
        raise ValueError(f"Unknown mode '{mode}' or missing template.")

    prompts = []

    # The rest is the same logic you used before, but simpler since template is already known:
    if mode == "subject_property":
        for combo in combinations:
            subj_leaf = combo['subject']['leaf']
            prop_leaf = combo['property']['leaf']

            subject_name = subj_leaf['name']
            subject_desc = subj_leaf.get('description', "")
            prop_name = prop_leaf['name']
            prop_desc = prop_leaf.get('description', "")

            prompt_text = prompt_template.format(
                model_name=model_name,
                guide=guide,
                subject_name=subject_name,
                subject_description=subject_desc,
                property_name=prop_name,
                property_description=prop_desc
            )
            prompts.append(prompt_text)

    elif mode == "entity_variation":
        for combo in combinations:
            subj_leaf = combo['subject']['leaf']
            entity = combo['entity']

            subject_name = subj_leaf['name']
            subject_desc = subj_leaf.get('description', "")
            ent_name = entity['name']
            ent_desc = entity.get('description', "")

            prompt_text = prompt_template.format(
                model_name=model_name,
                guide=guide,
                subject_name=subject_name,
                subject_description=subject_desc,
                entity_name=ent_name,
                entity_description=ent_desc
            )
            prompts.append(prompt_text)

    elif mode == "entity_placement":
        for combo in combinations:
            s1_leaf = combo['subject']['leaf']
            s2_leaf = combo['alt_subject']['leaf']
            entity = combo['entity']

            subject_name = s1_leaf['name']
            subject_desc = s1_leaf.get('description', "")
            alt_subject_name = s2_leaf['name']
            alt_subject_desc = s2_leaf.get('description', "")
            entity_name = entity['name']
            entity_desc = entity.get('description', "")

            prompt_text = prompt_template.format(
                model_name=model_name,
                guide=guide,
                subject_name=subject_name,
                subject_description=subject_desc,
                alt_subject_name=alt_subject_name,
                alt_subject_description=alt_subject_desc,
                entity_name=entity_name,
                entity_description=entity_desc
            )
            prompts.append(prompt_text)

    else:
        raise ValueError(f"Unknown mode for prompt building: {mode}")

    return prompts



def save_prompts(file_path, prompt_ids, prompts):
    """
    Saves prompt IDs and prompts into a JSON file.
    """
    data = []
    for pid, text in zip(prompt_ids, prompts):
        data.append({"id": pid, "prompt": text})
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_prompts(file_path):
    """
    Loads prompt IDs and prompts from a JSON file, returning a list of dicts:
    [
      {
        "id": "...",
        "prompt": "..."
      },
      ...
    ]
    """
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data

def reload_prompts_with_new_guide(
    input_frame,
    new_guide,
    taxonomy,
    output_tsv,
    model_name="SD",
    templates=TEMPLATES
):        
    all_leaves = get_leaves(taxonomy)
    print(f"Total leaves in taxonomy: {len(all_leaves)}", all_leaves[:2])
    leaves_dict = {l["leaf"]["name"]: l["leaf"]["description"] for l in all_leaves}

    # 4. Gather all entities from any leaf that has an 'entities' field
    entities = []
    for leaf_dict in all_leaves:
        node = leaf_dict['leaf']
        if 'entities' in node and isinstance(node['entities'], list):
            for entity in node['entities']:
                entities.append(entity)
                
    entities_dict = {l["name"]: l["description"] for l in entities}


    # Helper that converts a single row from the TSV into the "combo" structure
    def row_to_combo(row):
        mode = row["mode"]
        
        if type(row["property"]) == str:
            if not row["property"] in leaves_dict:
                leaves_dict[row["property"]] = find_class(taxonomy, row["property"])["description"]
            
        if type(row["subject"]) == str:
            if not row["subject"] in leaves_dict:
                leaves_dict[row["subject"]] = find_class(taxonomy, row["subject"])["description"]
            
        if type(row["alt_subject"]) == str:
            if not row["alt_subject"] in leaves_dict:
                leaves_dict[row["alt_subject"]] = find_class(taxonomy, row["alt_subject"])["description"]
        
        if mode == "subject_property":
            return {
                "subject": {
                    "leaf": {
                        "name": row["subject"],
                        "description": leaves_dict[row["subject"]]
                    }
                },
                "property": {
                    "leaf": {
                        "name": row["property"],
                        "description": leaves_dict[row["property"]]
                    }
                }
            }

        elif mode == "entity_variation":
            return {
                "subject": {
                    "leaf": {
                        "name": row["subject"],
                        "description": leaves_dict[row["subject"]]
                    }
                },
                "entity": {
                    "name": row["entity"],
                    "description": entities_dict[row["entity"]]
                }
            }

        elif mode == "entity_placement":
            return {
                "subject": {
                    "leaf": {
                        "name": row["subject"],
                        "description": leaves_dict[row["subject"]]
                    }
                },
                "alt_subject": {
                    "leaf": {
                        "name": row["alt_subject"],
                        "description": leaves_dict[row["alt_subject"]]
                    }
                },
                "entity": {
                    "name": row["entity"],
                    "description": entities_dict[row["entity"]]
                }
            }
        else:
            return {}

    # Build new prompt text for each row using the same 'build_pgp_prompts'
    def build_new_prompt(row):
        combo = row_to_combo(row)
        mode = row["mode"]

        # 'build_pgp_prompts' returns a list of prompts. We only have 1 combo, so:
        new_prompts = build_pgp_prompts(
            new_guide,          # the new guide text
            [combo],            # one combo per row
            model_name=model_name,
            mode=mode,
            templates=templates # reuse our global templates dict
        )
        return new_prompts[0] if new_prompts else row["prompt"]  # fallback if empty

    # Apply to each row in DataFrame
    df["prompt"] = df.apply(build_new_prompt, axis=1)

    return df

def compute_rows(combinations, prompts, mode, model_name):
    rows = []
    def get_leaves_from_combo(combo, mode):
        if mode == "subject_property":
            return {"subject": combo['subject']['leaf']['name'], "property": combo['property']['leaf']['name'], "alt_subject": None, "entity": None}
        elif mode == "entity_variation":
            return {"subject": combo['subject']['leaf']['name'], "property": None, "alt_subject": None, "entity": combo['entity']['name']}
        elif mode == "entity_placement":
            return {"subject": combo['subject']['leaf']['name'], "property": None, "alt_subject": combo['alt_subject']['leaf']['name'], "entity": combo['entity']['name']}
    for combo, prompt in zip(combinations, prompts):
        leaves = get_leaves_from_combo(combo, mode)
        row = {
            "mode": mode,
            "subject": leaves["subject"],
            "property": leaves["property"],
            "alt_subject": leaves["alt_subject"],
            "entity": leaves["entity"],
            "prompt": prompt,
            "model_name": model_name
        }
        rows.append(row)

    return rows

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prompt Generation Script")
    parser.add_argument("--model_name", type=str, default="FLUX", help="Model name to use in prompts")
    parser.add_argument("--mode", type=str, default="all", help="Mode to generate prompts for: subject_property, entity_variation, entity_placement, or all")
    parser.add_argument("--taxonomy_path", type=str, default="croc_syn/taxonomy.json", help="Path to the taxonomy JSON file")
    parser.add_argument("--output_path", type=str, default="outputs/croc_syn/prompt_gen/1_pre_prompts_flux.jsonl", help="Path to save the generated prompts TSV file")
    parser.add_argument("--secondary_model_name", type=str, default="SD", help="Secondary model name to use in prompts (if needed)")
    parser.add_argument("--secondary_output_path", type=str, default="outputs/croc_syn/prompt_gen/1_pre_prompts_sd.jsonl", help="Path to save the generated prompts TSV file with the secondary guide")
    args = parser.parse_args()

    print(f"Generating prompts in mode '{args.mode}'...")

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    os.makedirs(os.path.dirname(args.secondary_output_path), exist_ok=True)
    print(f"Output directories created for '{args.output_path}' and '{args.secondary_output_path}'.")

    modes = ["subject_property", "entity_variation", "entity_placement"] if args.mode == "all" else [args.mode]

    with open(GUIDE_DICT[args.model_name], "r", encoding="utf-8") as file:
        guide = file.read()

    with open(args.taxonomy_path, 'r', encoding='utf-8') as f:
        taxonomy = json.load(f)

    combinations = {m: get_combinations(taxonomy, mode=m) for m in modes}
    prompts = {m: build_pgp_prompts(guide, combinations[m], model_name=args.model_name, mode=m, templates=TEMPLATES) for m in modes}

    rows = []
    for mode in modes:
        rows.extend(compute_rows(combinations[mode], prompts[mode], mode, args.model_name))

    df = pd.DataFrame(rows, columns=["mode", "subject", "property", "alt_subject", "entity", "prompt", "model_name"])
    df.to_json(args.output_path, orient="records", lines=True, force_ascii=False)
    print(f"Saved all prompts to '{args.output_path}'.")


    df2 = reload_prompts_with_new_guide(input_frame=df, new_guide=open(GUIDE_DICT[args.secondary_model_name], "r", encoding="utf-8").read(), taxonomy=taxonomy, output_tsv=args.secondary_output_path, model_name=args.secondary_model_name, templates=TEMPLATES)
    df2.to_json(args.secondary_output_path, orient="records", lines=True, force_ascii=False)
    print(f"Reloaded prompts with new guide and saved to '{args.secondary_output_path}'.")

    # Print out row 1 of entifiy variation for both models
    print("\nExample prompt for entity_variation with FLUX:")
    print(df[df["mode"] == "entity_variation"].iloc[0]["prompt"])
    print("\nExample prompt for entity_variation with SD:")
    print(df2[df2["mode"] == "entity_variation"].iloc[0]["prompt"])