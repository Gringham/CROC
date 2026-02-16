# Evaluate on GenAI-Bench-Image (with 527 prompt) using a specific model
# Example scripts to run:
# VQAScore: python genai_image_eval.py --model clip-flant5-xxl
# CLIPScore: python genai_image_eval.py --model openai:ViT-L-14-336
# GPT4o VQAScore: python genai_image_eval.py --model gpt-4o
import argparse
import os
from metrics.CROCScore import CROCScore
from metrics.VQAScore import VQAScore
from metrics.Alignscore import Alignscore
#import t2v_metrics
from dataset import GenAIBench_Image
import json
import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm


def config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", default="./datasets", type=str,
                        help='Root directory for saving datasets.')
    parser.add_argument("--cache_dir", default="<CACHE_DIR>", type=str) 
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--batch_size", default=8, type=int)
    parser.add_argument("--num_prompts", default=1600, type=int, choices=[527, 1600])
    parser.add_argument("--model", default="crocscore", type=str, choices=["alignscore", "vqascore", "crocscore", "phiscore", "blip2itm"])
    parser.add_argument("--answer", default=None, type=str)
    parser.add_argument("--result_dir", default="./outputs/further_benchmarks/genaibench", type=str)
    parser.add_argument("--openai_key", default=None, type=str)
    parser.add_argument("--openai_key_path", default='./_OPENAI_API_KEY.txt', type=str)
    parser.add_argument("--top_logprobs", type=int, default=20)
    parser.add_argument("--detail", type=str, default='auto', choices=['low', 'auto', 'high'])
    return parser.parse_args()


tag_groups = {
    'basic': ['attribute', 'scene', 'spatial relation', 'action relation', 'part relation', 'basic'],
    'advanced': ['counting', 'comparison', 'differentiation', 'negation', 'universal', 'advanced'],
    'overall': ['basic', 'advanced', 'all']
}

def show_performance_per_skill(our_scores, dataset, items_name='images', prompt_to_items_name='prompt_to_images', print_std=False, tag_groups=tag_groups):
    tag_result = {}
    tag_file = f"{dataset.root_dir}/genai_skills.json"
    tags = json.load(open(tag_file))
    items = getattr(dataset, items_name)
    prompt_to_items = getattr(dataset, prompt_to_items_name)
    human_scores = [np.array(items[idx]['human_alignment']).mean() for idx in range(len(items))]
    items_by_model_tag = {}
    for tag in tags:
        items_by_model_tag[tag] = {}
        for prompt_idx in tags[tag]:
            for image_idx in prompt_to_items[f"{prompt_idx:05d}"]:
                model = items[image_idx]['model']
                if model not in items_by_model_tag[tag]:
                    items_by_model_tag[tag][model] = []
                items_by_model_tag[tag][model].append(image_idx)
    
    for tag in tags:
        # print(f"Tag: {tag}")
        tag_result[tag] = {}
        for model in items_by_model_tag[tag]:
            our_scores_mean = our_scores[items_by_model_tag[tag][model]].mean()
            our_scores_std = our_scores[items_by_model_tag[tag][model]].std()
            # print(f"{model} (Metric Score): {our_scores_mean:.2f} +- {our_scores_std:.2f}")
            human_scores_mean = np.array(human_scores)[items_by_model_tag[tag][model]].mean()
            human_scores_std = np.array(human_scores)[items_by_model_tag[tag][model]].std()
            # print(f"{model} (Human Score): {human_scores_mean:.1f} +- {human_scores_std:.1f}")
            tag_result[tag][model] = {
                'metric': {'mean': our_scores_mean, 'std': our_scores_std},
                'human': {'mean': human_scores_mean, 'std': human_scores_std},
            }
        # print()
        
    # print("All")
    tag_result['all'] = {}
    all_models = items_by_model_tag[tag]
    for model in all_models:
        all_model_indices = set()
        for tag in items_by_model_tag:
            all_model_indices = all_model_indices.union(set(items_by_model_tag[tag][model]))
        all_model_indices = list(all_model_indices)
        our_scores_mean = our_scores[all_model_indices].mean()
        our_scores_std = our_scores[all_model_indices].std()
        # print(f"{model} (Metric Score): {our_scores_mean:.2f} +- {our_scores_std:.2f}")
        human_scores_mean = np.array(human_scores)[all_model_indices].mean()
        human_scores_std = np.array(human_scores)[all_model_indices].std()
        # print(f"{model} (Human Score): {human_scores_mean:.1f} +- {human_scores_std:.1f}")
        tag_result['all'][model] = {
            'metric': {'mean': our_scores_mean, 'std': our_scores_std},
            'human': {'mean': human_scores_mean, 'std': human_scores_std},
        }
    
    for tag_group in tag_groups:
        for score_name in ['metric', 'human']:
            print(f"Tag Group: {tag_group} ({score_name} performance)")
            tag_header = f"{'Model':<20}" + " ".join([f"{tag:<20}" for tag in tag_groups[tag_group]])
            print(tag_header)
            for model_name in all_models:
                if print_std:
                    detailed_scores = [f"{tag_result[tag][model_name][score_name]['mean']:.2f} +- {tag_result[tag][model_name][score_name]['std']:.2f}" for tag in tag_groups[tag_group]]
                else:
                    detailed_scores = [f"{tag_result[tag][model_name][score_name]['mean']:.2f}" for tag in tag_groups[tag_group]]
                detailed_scores = " ".join([f"{score:<20}" for score in detailed_scores])
                model_scores = f"{model_name:<20}" + detailed_scores
                print(model_scores)
            print()
        print()



def main():
    args = config()
    if not os.path.exists(args.root_dir):
        os.makedirs(args.root_dir)
    os.makedirs(args.result_dir, exist_ok=True)

    dataset =  GenAIBench_Image(root_dir=args.root_dir, num_prompts=args.num_prompts) # 'num_prompts' is the number of prompts in GenAI-Bench（1600 in GenAI-Bench paper; 527 in VQAScore paper）
    
    result_path = f"{args.result_dir}/{args.model}_{args.num_prompts}_prompts.pt"

    if os.path.exists(result_path):
        print(f"Result file {result_path} already exists. Skipping.")
        scores = torch.load(result_path)
    else:
        if args.model == "alignscore":
            scorer = Alignscore()
        elif args.model == "vqascore":
            scorer = VQAScore()
        elif args.model == "crocscore":
            scorer = CROCScore(batch_size=4)
        elif args.model == "phiscore":
            scorer = CROCScore(model_name="microsoft/Phi-4-multimodal-instruct", batch_size=4)
        elif args.model == "blip2itm":
            scorer = VQAScore(base="blip2itm")
        
        num_samples = len(dataset)
        num_images = len(dataset[0]['images'])
        num_texts = len(dataset[0]['texts'])
        scores = torch.zeros(num_samples, num_images, num_texts).to(args.device)
        
        dataloader = DataLoader(dataset, batch_size=scorer.batch_size, shuffle=False)
        counter = 0
        for batch_idx, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
            cur_batch_size = len(batch['images'][0])
            assert len(batch['images']) == num_images, \
                f"Number of image options in batch {batch_idx} is {len(batch['images'])}. Expected {num_images} images."
            assert len(batch['texts']) == num_texts, \
                f"Number of text options in batch {batch_idx} is {len(batch['texts'])}. Expected {num_texts} texts."
            
            for image_idx in range(num_images):
                images = batch['images'][image_idx]
                for text_idx in range(num_texts):
                    texts = batch['texts'][text_idx]
                    #scores = scorer(texts, images)
                    print(texts, images)

                    scores[counter:counter+cur_batch_size, image_idx, text_idx] = torch.as_tensor(scorer(texts, images), dtype=torch.float32, device=scores.device)
            counter += cur_batch_size
        
        torch.save(scores, result_path)
        
    ### Get performance per skill
    our_scores = scores.mean(axis=1)
    show_performance_per_skill(our_scores, dataset, print_std=True)    
    
    print("Overall Alignment Performance")
    ### Overall Alignment performance
    dataset.evaluate_scores(scores)

    ### Alignment performance per skill
    print("Evaluating scores of each skill for model:", args.model)
    skill_result = dataset.evaluate_scores_per_skill(scores)
    print("Results saved to:", f"{args.result_dir}/{args.model}_{args.num_prompts}_per_skill.json")
    output_file = f"{args.result_dir}/{args.model}_{args.num_prompts}_per_skill.json"
    with open(output_file, 'w') as f:
        json.dump(skill_result, f)
    print("\n")


if __name__ == "__main__":
    main()
