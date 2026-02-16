# Read in the human annotations from Tifa and from DSG

import json
import os
from pydoc import text
import numpy as np
import pandas as pd
import torch
import argparse
from metrics.Alignscore import Alignscore
from metrics.VQAScore import VQAScore
from metrics.CROCScore import CROCScore

if __name__ == "__main__":
    parsr = argparse.ArgumentParser()
    parsr.add_argument("--metric", type=str, choices=["alignscore", "vqascore", "crocscore", "blip2itm", "phiscore"], default="crocscore")
    parsr.add_argument("--output_path", type=str, default="outputs/further_benchmarks/tifa")
    args = parsr.parse_args()

    os.makedirs(args.output_path, exist_ok=True)

    if args.metric == "alignscore":
        metric = Alignscore()
        kwargs = {"direct_img": False}
    elif args.metric == "vqascore":
        metric = VQAScore()
        kwargs = {}
    elif args.metric == "crocscore":
        metric = CROCScore(batch_size=4)
        kwargs = {"direct_img": False}
    elif args.metric == "phiscore":
        metric = CROCScore(model_name="microsoft/Phi-4-multimodal-instruct", batch_size=4)
        kwargs = {"direct_img": False}
    elif args.metric == "blip2itm":
        metric = VQAScore(base="blip2itm")
        kwargs = {}


    with open("further_benchmarks/tifa/human_annotations/human_annotations_with_scores.json") as f:
        tifa_annotations_original = json.load(f)
        
    dsg_path = "further_benchmarks/DSG/dsg/data/tifa160-likert-anns.csv"
    df = pd.read_csv(dsg_path)
    # Group by item_id and calculate the average "answer" for each group.
    from collections import Counter

    def most_frequent_or_none(x):
        if x.nunique() == 1:
            return x.iloc[0]
        else:
            # Return the most frequent value, or None if ambiguous
            most_common = Counter(x).most_common(1)
            return most_common[0][0] if most_common else None

    dsg_annotations = df.groupby(["item_id", "t2i_model"]).agg(
        answer_mean=("answer", "mean"),
        **{col: (col, most_frequent_or_none) for col in df.select_dtypes(include="object").columns if col != "item_id"}
    )

    base = "further_benchmarks/tifa/images/annotated_images"
    filenames = os.listdir(base)
    image_keys = [s.split(".")[0] for s in filenames]

    # Add a new column to dsg_annotations for the image paths (base + "source_id" column + _ + "t2i_model" column with _ instead of - + ".jpg"
    dsg_annotations["image_path"] = base + "/" + dsg_annotations["source_id"] + "_" + dsg_annotations["t2i_model"].str.replace("-", "_") + ".jpg"
    dsg_annotations["key"] = dsg_annotations["source_id"] + "_" + dsg_annotations["t2i_model"].str.replace("-", "_")

    # Replace "sd1dot1" with "stable_diffusion_v1_1" in the key column. Likewise 2dot1 and 1dot5
    dsg_annotations["key"] = dsg_annotations["key"].str.replace("sd1dot1", "stable_diffusion_v1_1")
    dsg_annotations["key"] = dsg_annotations["key"].str.replace("sd2dot1", "stable_diffusion_v2_1")
    dsg_annotations["key"] = dsg_annotations["key"].str.replace("sd1dot5", "stable_diffusion_v1_5")

    image_paths = [os.path.join(base, d) for d in filenames]
        
    gt_scores_original=[tifa_annotations_original[k]["human_avg"] for k in image_keys]
    texts_original = [tifa_annotations_original[k]["text"] for k in image_keys]    

    # Get the dsg ground truth scores ordered by image_keys
    gt_scores_dsg = []
    for key in image_keys:
        gt_scores_dsg.append(dsg_annotations[dsg_annotations["key"] == key]["answer_mean"].values[0].item())


    scores = metric(texts_original, image_paths)

    print(scores)
    print(gt_scores_original)

    # Calculate kendall and pearson correlation
    from scipy.stats import kendalltau, pearsonr
    kendall_corr = kendalltau(gt_scores_original, scores).correlation
    pearson_corr = pearsonr(gt_scores_original, scores)[0]

    res_dict = {
        "Original Tifa Annotations": {
            "Kendall Correlation": kendall_corr,
            "Pearson Correlation": pearson_corr
        },
        "DSG Annotations": {
            "Kendall Correlation": kendalltau(gt_scores_dsg, scores).correlation,
            "Pearson Correlation": pearsonr(gt_scores_dsg, scores)[0]
        }
    }
    print(res_dict)

    with open(f"{args.output_path}/{args.metric}_results.json", "w") as f:
        json.dump(res_dict, f, indent=4)