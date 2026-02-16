#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Full pipeline:
    • load metric TSVs
    • compute Forward/Inverse × Text/Image scores
    • render four heat-maps (–1…+1 scale) with an extra “avg” column
      – one shared colour-bar (outside grid)
      – x-ticks hidden on first row, y-ticks hidden on right column
      – titles: Text, Text_Inv, Img, Img_Inv
    • global font family: Times
"""

import os
from pathlib import Path
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
from tqdm import tqdm

# ---------------------------------------------------------------------
# GLOBAL STYLE  (Times everywhere)
# ---------------------------------------------------------------------
mpl.rcParams["font.family"] = "serif"
mpl.rcParams["font.serif"]  = ["Times New Roman", "Times", "DejaVu Serif"]

# ---------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------
root_dir   = Path(".")       # adjust if TSVs are elsewhere

categories = [
    "shapes", "spacial_relation", "size_relation", "counting", "body_parts",
"action", "things_parts", "negation"
]

metrics_raw = [
    "PickScore" #"Blip2ITM", "CLIPScore_Large", "SSAlign", "BVQA", "VQAScore", "VQAScore_4o", "CROCScore_No_Tune", "CROCScore"
]

directions = ["F_text", "I_text", "F_image", "I_image"]

img_model_occurrences = {
    "flux": 0,
    "sd": 0,
    "gpt": 0,
}

# ---------------------------------------------------------------------
# LOAD + AGGREGATE
# ---------------------------------------------------------------------
raw_scores = {cat: {} for cat in categories}

all_all_filenames = []
for cat in tqdm(categories, desc="Categories"):
    # ---------- PROMPT TSVs ----------
    p1 = root_dir / f"outputs/croc_hum/metrics/scores/{cat}_prompt_img_scores.tsv"
    c1 = root_dir / f"outputs/croc_hum/metrics/scores/{cat}_contrast_img_scores.tsv"

    prompt_df  = pd.read_csv(p1, sep="\t")  
    contrast_df  = pd.read_csv(c1, sep="\t")


    def get_img_gen_model(row, rowname="img_paths_prompts"):
        parts = row[rowname]
        if "flux" in parts.lower():
            return "flux"
        elif "stable_diffusion" in parts.lower():
            return "sd"
        else:
            return "gpt"
        

    prompt_df["img_gen_model"] = prompt_df.apply(lambda row: get_img_gen_model(row, "prompt_paths"), axis=1)
    contrast_df["img_gen_model"] = contrast_df.apply(lambda row: get_img_gen_model(row, "contrast_paths"), axis=1)


    prompt_model_counts = prompt_df["img_gen_model"].value_counts().to_dict()
    contrast_model_counts = contrast_df["img_gen_model"].value_counts().to_dict()
    for model in img_model_occurrences.keys():
        img_model_occurrences[model] += prompt_model_counts.get(model, 0)
        img_model_occurrences[model] += contrast_model_counts.get(model, 0)

    # add _compared flags
    for m in metrics_raw:
        prompt_df[f"{m}_compared"]   = prompt_df[f"{m}___orig_text___orig_img"] > prompt_df[f"{m}___contrast_text___orig_img"] #+ thresh
        contrast_df[f"{m}_compared"] = contrast_df[f"{m}___contrast_text___contrast_img"] > contrast_df[f"{m}___orig_text___contrast_img"] #+ thresh

    p_group = prompt_df.groupby("id").mean(numeric_only=True)
    c_group = contrast_df.groupby("id").mean(numeric_only=True)

    print(len(prompt_df), len(contrast_df))

    # aggregate four numbers per metric
    for m in metrics_raw:
        F_text = p_group[f"{m}_compared"].mean()
        I_text = c_group[f"{m}_compared"].mean()

        print(f"Category: {cat}, Metric: {m}, F_text: {F_text:.4f}, I_text: {I_text:.4f}")

        common = set(prompt_df["id"]) & set(contrast_df["id"])
        f_vals, i_vals = [], []

        for idx in common:
            pg = prompt_df[prompt_df["id"] == idx]
            cg = contrast_df[contrast_df["id"] == idx]

            f_hi = pg[f"{m}___orig_text___orig_img"].to_numpy()
            f_lo = cg[f"{m}___orig_text___contrast_img"].to_numpy()
            i_hi = cg[f"{m}___contrast_text___contrast_img"].to_numpy()
            i_lo = pg[f"{m}___contrast_text___orig_img"].to_numpy()

            f_vals.append((f_hi[:, None] > f_lo ).mean()) # + thresh
            i_vals.append((i_hi[:, None] > i_lo ).mean()) # + thresh

        raw_scores[cat][m] = [F_text, I_text, float(np.mean(f_vals)), float(np.mean(i_vals))]

# ---------------------------------------------------------------------
# LONG-FORM DATA
# ---------------------------------------------------------------------
df = pd.DataFrame(
    {"Category": cat, "Metric": met, "Direction": d, "Score": v}
    for cat, m_dict in raw_scores.items()
    for met, vals in m_dict.items()
    for d, v in zip(directions, vals)
)

df["Category"] = df["Category"].replace({"spacial_relation": "spatial_relation"})
df["Metric"]   = df["Metric"].replace({
    "SSAlign": "AlignScore",
    "Blip2ITM": "BLIP2-ITM",
    "BVQA": "BVQA",
    "CLIPScore_Large": "CLIPScore",
    "CROCScore_No_Tune": "CROCScore_P",
    "CROCScore_2310-750": "CROCScore",
    "VQAScore_4o": "VQAScore_4o*"
})

# ---------------------------------------------------------------------
# HEAT-MAPS (with per-metric “avg” column)
# ---------------------------------------------------------------------
df_h = df.copy()
df_h["Score"] = df_h["Score"] * 2 - 1          # 0…1 → –1…+1

cats_order     = sorted(df_h["Category"].unique())
cats_order_ext = cats_order + ["AVG"]          # include "avg" at the end
metrics_order  = (
    ["AlignScore", "CLIPScore", "BLIP2-ITM","PickScore", "BVQA", "VQAScore_4o*", "VQAScore",
     "CROCScore_P", "CROCScore"]
)

title_map = {
    "F_text":  "Text",
    "I_text":  "Text_Inv",
    "F_image": "Img",
    "I_image": "Img_Inv"
}

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharex=True, dpi=200)
groups = [["F_text","I_text"], ["F_image","I_image"]]

#axes = axes.ravel()

vmin, vmax      = -1, 1
cmap            = "turbo_r"
shared_mappable = None

for idx, (ax, grp) in enumerate(zip(axes, groups)):
    # compute the two mats and average them
    mats = [
        df_h[df_h["Direction"] == d]
            .pivot(index="Metric", columns="Category", values="Score")
            .reindex(index=metrics_order)
        for d in grp
    ]
    mat = (mats[0] + mats[1]) / 2

    mat = mat.T
    mat["AVG"] = mat.mean(axis=1)
    mat = mat.T

    # compute the avg column and then reorder columns
    mat["AVG"] = mat.mean(axis=1)
    mat = mat.reindex(columns=cats_order_ext)   # order cols + avg

    # draw
    hm = sns.heatmap(
        mat, ax=ax, cmap=cmap,
        vmin=vmin, vmax=vmax,
        cbar=False,
        linewidths=.25, linecolor="white",
        annot=True, fmt=".2f",
        annot_kws={"size": 12.5}
    )

    ax.set_title("Avg " + ("Text-Based" if idx == 0 else "Image-Based"), pad=6, fontsize=10)
    ax.set_xlabel("")
    ax.set_ylabel("")

    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=10)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=10)

    #if idx < 2:  # hide x-ticks on first row
    #ax.tick_params(axis="x", bottom=False, labelbottom=False)
    if idx % 2 == 1:  # hide y-ticks on right column
        ax.tick_params(axis="y", left=False, labelleft=False)

    if shared_mappable is None:
        shared_mappable = hm

# shared colour-bar (outside grid)
cbar_ax = fig.add_axes([0.89, 0.30, 0.02, 0.40])
fig.colorbar(shared_mappable.collections[0], cax=cbar_ax, label="Score (–1 to 1)")

plt.tight_layout(rect=[0, 0, 0.88, 1])

# mkdir if not exists
os.makedirs("outputs/croc_hum/plots", exist_ok=True)
plt.savefig("outputs/croc_hum/plots/metric_heatmaps.pdf", dpi=300)
plt.show()

print(len(all_all_filenames))

print("Image generation model occurrences:")
for model, count in img_model_occurrences.items():
    print(f"{model}: {count}")