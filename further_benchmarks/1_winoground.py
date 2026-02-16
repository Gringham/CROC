from datasets import load_dataset
from metrics.Alignscore import Alignscore
from metrics.VQAScore import VQAScore
from metrics.CROCScore import CROCScore

import json
import argparse
import os



if __name__ == "__main__":
    parsr = argparse.ArgumentParser()
    parsr.add_argument("--metric", type=str, choices=["alignscore", "vqascore", "crocscore", "blip2itm", "phiscore"], default="crocscore")
    parsr.add_argument("--output_path", type=str, default="outputs/further_benchmarks/winoground")
    args = parsr.parse_args()
    examples = load_dataset('facebook/winoground', token="<TOKEN>")["test"]
    

    os.makedirs(args.output_path, exist_ok=True)

    if args.metric == "alignscore":
        metric = Alignscore()
        kwargs = {"direct_img": True}
    elif args.metric == "vqascore":
        metric = VQAScore()
        kwargs = {}
    elif args.metric == "crocscore":
        metric = CROCScore(batch_size=4)
        kwargs = {"direct_img": True}
    elif args.metric == "phiscore":
        metric = CROCScore(model_name="microsoft/Phi-4-multimodal-instruct", batch_size=4)
        kwargs = {"direct_img": True}
    elif args.metric == "blip2itm":
        metric = VQAScore(base="blip2itm")
        kwargs = {}

    # Match 1
    m1_scores = metric(examples["caption_0"], examples["image_0"], **kwargs)

    # Match 2
    m2_scores = metric(examples["caption_1"], examples["image_1"], **kwargs)

    # Non-Match 1
    n1_scores = metric(examples["caption_1"], examples["image_0"], **kwargs)

    # Non-Match 2
    n2_scores = metric(examples["caption_0"], examples["image_1"], **kwargs)

    accuracy = lambda scores_1, scores_2 : [s1 > s2  for s1, s2 in zip(scores_1, scores_2)]

    ft_acc = accuracy(m1_scores, n1_scores)
    it_acc = accuracy(m2_scores, n2_scores)
    fi_acc = accuracy(m1_scores, n2_scores)
    ii_acc = accuracy(m2_scores, n1_scores)

    text_acc = [a and b for a, b in zip(ft_acc, it_acc)]
    img_acc = [a and b for a, b in zip(fi_acc, ii_acc)]
    group_acc = [a and b for a, b in zip(text_acc, img_acc)]

    res_dict = {
        "Forward Text Based Accuracy": sum(ft_acc)/len(ft_acc),
        "Inverse Text Based Accuracy": sum(it_acc)/len(it_acc),
        "Forward Img Based Accuracy": sum(fi_acc)/len(fi_acc),
        "Inverse Img Based Accuracy": sum(ii_acc)/len(ii_acc),
        "Text Accuracy": sum(text_acc)/len(text_acc),
        "Image Accuracy": sum(img_acc)/len(img_acc),
        "Group Accuracy": sum(group_acc)/len(group_acc)
    }

    print(res_dict)
    
    with open(f"{args.output_path}/{args.metric}_results.json", "w") as f:
        json.dump(res_dict, f, indent=4)