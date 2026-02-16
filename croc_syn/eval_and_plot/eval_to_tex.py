import os
import pandas as pd
from scipy.stats import kendalltau
import tqdm
import argparse

# --- Lookup dictionaries ---
setup_mapping = {
    "1_1_inverse": "Text_Inv",
    "1_1": "Text",
    "2_1_inverse": "Img_Inv",
    "2_1": "Img"
}

metric_mapping = {
    "BVQA": "BVQA",
    "VQAScore": "VQA",
    "CLIPScore": "CLIP",
    "SSAlign": "Align",
    "SSBlip": "Blip",
    "PickScore": "Pick",
    "RandomScore": "Random",
}

exp_mapping = {
    "entity_variation": "EV",
    "entity_placement": "EP",
    "subject_property": "PV"
}

def escape_latex(s):
    """Simple escape function for underscores."""
    return s.replace("_", r"\_")

#############################################################################
#                   HELPER FUNCTIONS
#############################################################################
def combine_duplicate_columns(df):
    """Combine duplicate columns by taking the first non-null value per row."""
    combined = {}
    for col in df.columns.unique():
        sub_df = df.loc[:, df.columns == col]
        if sub_df.shape[1] > 1:
            combined[col] = sub_df.bfill(axis=1).iloc[:, 0]
        else:
            combined[col] = sub_df.iloc[:, 0]
    return pd.DataFrame(combined)

def add_avg_row_col(df):
    """Add an average row and an average column."""
    df_avg = df.copy()
    df_avg['Average'] = df_avg.mean(axis=1)
    avg_row = df_avg.mean(axis=0)
    df_avg.loc['Average'] = avg_row
    return df_avg

def format_value(val, col_max):
    """Round to 3 decimals, color blue if >0.5, red otherwise; bold if equal to maximum."""
    try:
        numeric_val = float(val)
    except:
        return str(val)
    formatted = f"{numeric_val:.3f}"

    if abs(numeric_val - col_max) < 1e-9:
        formatted = r"\textbf{" + formatted + "}"
    return formatted

def escape_underscores_in_index_and_columns(df):
    """Escape underscores in both index and column names for LaTeX."""
    if isinstance(df.index, pd.MultiIndex):
        new_levels = []
        for level in df.index.levels:
            new_levels.append([str(x).replace("_", r"\_") for x in level])
        df.index = df.index.set_levels(new_levels)
    else:
        df.index = [str(x).replace("_", r"\_") for x in df.index]
    df.columns = [str(x).replace("_", r"\_") for x in df.columns]
    return df

def abbreviate_column(col):
    """
    Abbreviate a column header.
    If the column ends with '_compared', remove that suffix and look up in metric_mapping.
    """
    base = col[:-len("_compared")] if col.endswith("_compared") else col
    return metric_mapping.get(base, base)

def save_and_print_latex_table(df, filename, caption=None, label=None):
    """Generate a LaTeX table from df with formatted numeric values and escaping.
       This version bolds the highest numeric value in each row."""
    formatted_df = df.copy()
    
    # Convert all columns to numeric when possible
    numeric_df = formatted_df.apply(pd.to_numeric, errors='coerce')
    # Compute maximum per row (ignoring NaNs)
    row_max = numeric_df.max(axis=1)
    
    def format_cell(x, idx):
        # If the cell is NaN, return an empty string
        if pd.isna(x):
            return ""
        try:
            numeric_val = float(x)
        except Exception:
            return str(x)
        formatted = f"{numeric_val:.3f}"
        # Bold if this value is equal to the maximum in its row (within a small tolerance)
        if abs(numeric_val - row_max.loc[idx]) < 1e-9:
            formatted = r"\textbf{" + formatted + "}"
        return formatted

    # Apply the formatting for each cell, row by row
    for col in formatted_df.columns:
        formatted_df[col] = [format_cell(formatted_df.at[idx, col], idx) 
                             for idx in formatted_df.index]
    
    # Escape underscores in both the index and column names for LaTeX
    formatted_df = escape_underscores_in_index_and_columns(formatted_df)
    
    output_folder = "0_control_outputs/latex_tables"
    os.makedirs(output_folder, exist_ok=True)
    latex_tabular = formatted_df.to_latex(bold_rows=True, escape=False)
    cap = escape_latex(caption) if caption else ""
    latex_str = (
        r"\begin{table}[htbp]" "\n" +
        r"\centering" "\n" +
        latex_tabular + "\n" +
        (r"\caption{" + cap + "}" "\n" if cap else "") +
        (r"\label{" + label + "}" "\n" if label else "") +
        r"\end{table}"
    )
    print(latex_str)
    with open(os.path.join(output_folder, filename), "w") as f:
        f.write(latex_str)


def create_large_combined_table(df_a, df_b, df_c, df_d,
                                title_a, title_b, title_c, title_d,
                                output_file, caption=None, label=None):
    """
    Combine four DataFrames into one large LaTeX 2×2 table.
    Adds "Random" as its own separate column group.
    Appends a final 'Average' ROW with per-column means across all rows.
    This method is largely generated
    """
    # Concatenate using provided titles as keys.
    df_comb = pd.concat([df_a, df_b, df_c, df_d],
                        keys=[title_a, title_b, title_c, title_d])

    # Abbreviate the raw column headers.
    df_comb.columns = [abbreviate_column(col) for col in df_comb.columns]

    # ---- GROUPING COLUMNS ----
    all_cols = list(df_comb.columns)
    grouped = {"Embedding-Based": [], "Fine-tuned": [], "VQA-Based": [], "Random": []}
    others = []
    for col in all_cols:
        if col in {"CLIP", "Align", "Blip", "CLIPScore_Large", "Blip2ITM"}:
            grouped["Embedding-Based"].append(col)
        elif col == "Pick":
            grouped["Fine-tuned"].append(col)
        elif col in {"BVQA", "VQA"}:
            grouped["VQA-Based"].append(col)
        elif col == "Random":
            grouped["Random"].append(col)
        else:
            others.append(col)
    # Sort within groups and others
    for grp in grouped:
        grouped[grp].sort()
    others.sort()
    # Final column order: embedding, fine-tuned, VQA, random, others
    new_order = (
        grouped["Embedding-Based"] + grouped["Fine-tuned"] +
        grouped["VQA-Based"] + grouped["Random"] + others
    )
    df_comb = df_comb[new_order]

    # compute final Average ROW (per column)
    def _parse_cell_to_pair(cell):
        """Return (v1, v2, had_split) where v1/v2 may be None."""
        if isinstance(cell, str) and " / " in cell:
            parts = cell.split(" / ")
            had_split = True
            try:
                v1 = float(parts[0])
            except:
                v1 = None
            try:
                v2 = float(parts[1])
            except:
                v2 = None
            return v1, v2, had_split
        else:
            try:
                v = float(cell)
                return v, v, False
            except:
                return None, None, False

    avg_row_vals = {}
    for col in new_order:
        v1s, v2s = [], []
        saw_split = False
        for _, cell in df_comb[col].items():
            v1, v2, had_split = _parse_cell_to_pair(cell)
            saw_split = saw_split or had_split
            if v1 is not None:
                v1s.append(v1)
            if v2 is not None:
                v2s.append(v2)
        avg1 = (sum(v1s) / len(v1s)) if v1s else None
        avg2 = (sum(v2s) / len(v2s)) if v2s else None
        if saw_split or (avg1 is not None and avg2 is not None and abs(avg1 - avg2) > 1e-12):
            s1 = "-" if avg1 is None else f"{avg1:.3f}"
            s2 = "-" if avg2 is None else f"{avg2:.3f}"
            avg_row_vals[col] = f"{s1} / {s2}"
        else:
            # Use a single numeric value if possible; otherwise a dash.
            if avg1 is not None:
                avg_row_vals[col] = avg1
            elif avg2 is not None:
                avg_row_vals[col] = avg2
            else:
                avg_row_vals[col] = "-"
    # Choose an index key for the Average row that matches the index structure.
    if isinstance(df_comb.index, pd.MultiIndex):
        avg_row_key = tuple(["Average"] + [""] * (df_comb.index.nlevels - 1))
    else:
        avg_row_key = "Average"
    # Append Average row to the raw combined dataframe.
    df_comb.loc[avg_row_key] = pd.Series(avg_row_vals)

    # ---- FORMAT ROWS FOR COMBINED CELLS ----
    def format_row_combined(row):
        # Disable bolding for the Average row.
        is_avg_row = (row.name == avg_row_key)

        # For each cell that is combined, parse the two numbers.
        model1_vals = {}
        model2_vals = {}
        if not is_avg_row:
            for col in row.index:
                cell = row[col]
                if isinstance(cell, str) and " / " in cell:
                    parts = cell.split(" / ")
                    try:
                        v1 = float(parts[0])
                    except:
                        v1 = None
                    try:
                        v2 = float(parts[1])
                    except:
                        v2 = None
                    model1_vals[col] = v1
                    model2_vals[col] = v2
                else:
                    try:
                        val = float(cell)
                        model1_vals[col] = val
                        model2_vals[col] = val
                    except:
                        model1_vals[col] = None
                        model2_vals[col] = None
        valid_model1 = [v for v in model1_vals.values() if v is not None]
        valid_model2 = [v for v in model2_vals.values() if v is not None]
        max1 = (max(valid_model1) if valid_model1 else None) if not is_avg_row else None
        max2 = (max(valid_model2) if valid_model2 else None) if not is_avg_row else None

        formatted_row = {}
        for col in row.index:
            cell = row[col]
            if isinstance(cell, str) and " / " in cell:
                parts = cell.split(" / ")
                try:
                    v1 = float(parts[0])
                    s1 = f"{v1:.3f}"
                    if (max1 is not None) and abs(v1 - max1) < 1e-9:
                        s1 = r"\textbf{" + s1 + "}"
                except:
                    s1 = parts[0]
                try:
                    v2 = float(parts[1])
                    s2 = f"{v2:.3f}"
                    if (max2 is not None) and abs(v2 - max2) < 1e-9:
                        s2 = r"\textbf{" + s2 + "}"
                except:
                    s2 = parts[1]
                formatted_row[col] = s1 + " / " + s2
            else:
                try:
                    val = float(cell)
                    formatted_str = f"{val:.3f}"
                    if (max1 is not None) and abs(val - max1) < 1e-9:
                        formatted_str = r"\textbf{" + formatted_str + "}"
                    formatted_row[col] = formatted_str
                except:
                    formatted_row[col] = str(cell)
        return pd.Series(formatted_row)

    formatted_df = df_comb.copy()
    formatted_df = formatted_df.apply(format_row_combined, axis=1)
    # ---------------------------------------

    # Rebuild the MultiIndex for the rows.
    raw_index = df_comb.index
    if isinstance(raw_index, pd.MultiIndex):
        new_index_tuples = []
        for tup in raw_index:
            first_val = tup[0]
            if isinstance(first_val, str) and first_val.startswith("Metrics (") and first_val.endswith(")"):
                first_val = first_val[len("Metrics ("):-1]
            new_first = setup_mapping.get(first_val, first_val)
            if len(tup) >= 2:
                new_second = exp_mapping.get(tup[1], tup[1])
                new_tuple = (new_first, new_second) + tup[2:]
            else:
                new_tuple = (new_first,) + tup[1:]
            new_index_tuples.append(new_tuple)
        if len(new_index_tuples[0]) == 2:
            names_list = ["Setup", "Prompt"]
        else:
            names_list = ["Setup", "Prompt"] + [f"Level {i}" for i in range(3, len(new_index_tuples[0])+1)]
        formatted_df.index = pd.MultiIndex.from_tuples(new_index_tuples, names=names_list)
    else:
        formatted_df.index = [setup_mapping.get(x, x) for x in raw_index]
        formatted_df.index.name = "Setup"

    # Optionally bold the 'Average' label in the first index level.
    try:
        if isinstance(formatted_df.index, pd.MultiIndex):
            idx = list(formatted_df.index)
            avg_pos = idx.index(tuple(["Average"] + [""] * (formatted_df.index.nlevels - 1)))
            # Replace first level value with bold text for display
            new_tuple = (r"\textbf{Average}",) + idx[avg_pos][1:]
            idx[avg_pos] = new_tuple
            formatted_df.index = pd.MultiIndex.from_tuples(idx, names=formatted_df.index.names)
        else:
            idx = list(formatted_df.index)
            avg_pos = idx.index("Average")
            idx[avg_pos] = r"\textbf{Average}"
            formatted_df.index = pd.Index(idx, name=formatted_df.index.name)
    except ValueError:
        # If for some reason 'Average' not found, skip styling silently.
        pass

    # Escape underscores in index and columns.
    formatted_df = escape_underscores_in_index_and_columns(formatted_df)

    # ---- BUILD COLUMN FORMAT WITH VERTICAL LINES BETWEEN GROUPS ----
    # Determine number of index columns.
    if isinstance(formatted_df.index, pd.MultiIndex):
        num_index = formatted_df.index.nlevels
    else:
        num_index = 1
    index_fmt = "l" * num_index

    # Helper to determine the metric group.
    def get_group(col):
        if col in {"CLIP", "Align", "Blip", "CLIPScore_Large"}:
            return "Embedding-Based"
        elif col == "Pick":
            return "Fine-tuned"
        elif col in {"BVQA", "VQA"}:
            return "VQA-Based"
        elif col == "Random":
            return "Random"
        else:
            return ""

    metric_cols = list(formatted_df.columns)
    metric_fmt = ""
    if metric_cols:
        metric_fmt = "l"
        for i in range(1, len(metric_cols)):
            if get_group(metric_cols[i]) != get_group(metric_cols[i-1]):
                metric_fmt += "|" + "l"
            else:
                metric_fmt += "l"
    col_format = index_fmt + "|" + metric_fmt
    # --------------------------------------------------------

    # Generate LaTeX tabular code.
    latex_tabular = formatted_df.to_latex(multicolumn=True, multirow=True,
                                          escape=False, column_format=col_format)

    # ---- INSERT HEADER ROWS ----
    lines = latex_tabular.splitlines()
    toprule_idx = None
    for i, line in enumerate(lines):
        if r"\toprule" in line:
            toprule_idx = i
            break
    if toprule_idx is not None:
        # First header row: overall "Metrics" label spanning all metric columns.
        metrics_header = r"\multicolumn{" + str(num_index) + r"}{c}{}"
        if len(metric_cols) > 0:
            metrics_header += " & " + r"\multicolumn{" + str(len(metric_cols)) + r"}{c}{Metrics}"
        metrics_header += r" \\"
        lines.insert(toprule_idx+1, metrics_header)

        # Second header row: grouping columns into Embedding-Based, Fine-tuned, VQA-Based, Random.
        groups = []
        current_group = None
        count = 0
        for col in metric_cols:
            group = get_group(col)
            if group == current_group:
                count += 1
            else:
                if current_group is not None:
                    groups.append((current_group, count))
                current_group = group
                count = 1
        groups.append((current_group, count))
        group_header = r"\multicolumn{" + str(num_index) + r"}{c}{}"
        for group, span in groups:
            group_label = group if group != "" else ""
            group_header += " & " + r"\multicolumn{" + str(span) + r"}{c}{" + group_label + r"}"
        group_header += r" \\"
        lines.insert(toprule_idx+2, group_header)
        latex_tabular = "\n".join(lines)
    # -----------------------------

    # Build dynamic caption.
    setup_desc = ", ".join([f"{escape_latex(abbr)} = {escape_latex(key)}" for key, abbr in setup_mapping.items()])
    exp_desc = "ev = entity\\_variation, ep = entity\\_placement, pv = subject\\_property"
    metric_desc = ", ".join([f"{escape_latex(abbr)} = {escape_latex(key)}" for key, abbr in metric_mapping.items()])
    general_caption = (
        "This table summarizes the evaluation metrics. "
        "The first column ('Setup') indicates the evaluation type (abbreviated as: " +
        setup_desc + "). "
        "The second column ('Exp') denotes the experiment (abbreviated as: " + exp_desc + "). "
        "Metric abbreviations in the table: " + metric_desc + ". "
        "The last row ('Average') reports the mean for each metric across all rows."
    )
    cap = escape_latex(caption) + " " + general_caption if caption else general_caption

    latex_str = (
        r"\begin{table*}[htbp]" "\n" +
        r"\centering" "\n" +
        latex_tabular + "\n" +
        (r"\caption{" + cap + "}" "\n" if cap else "") +
        (r"\label{" + label + "}" "\n" if label else "") +
        r"\end{table*}"
    )

    print(latex_str)
    with open(output_file, "w") as f:
        f.write(latex_str)

def scale(s, p):
    if s <= p:
        return (s  - p) / p
    return (s - p) / (1 - p)

def combine_dfs(df1, df2, mode=0):
    """
    Combines the two Dataframes (per model) cell by cell and scale according to mode
    """
    combined = df1.copy()
    for idx in df1.index: # each row
        for col in df1.columns: # each column 
            v1 = float(df1.loc[idx, col]) 
            v2 = float(df2.loc[idx, col])
            score = (v1+v2)/2 # average the cells of the two models

            if mode == 0:
                combined.loc[idx, col] = scale(score, 5/6) 

            if mode == 1:
                combined.loc[idx, col] = scale(score, 0.5)

            combined.loc[idx, col] = round(combined.loc[idx, col], 3)

    return combined

def print_setup_stats(df, setup, id_col="prompt_id"):
    """Compute and return a dictionary of average metrics grouped by mode."""
    stats = {}
    for m in ["Blip2ITM", "BVQA", "VQAScore", "CLIPScore_Large", "SSAlign", "PickScore"]:
        if setup == "forward_text": 
            higher = f"{m}___orig_text___orig_img"
            lower = f"{m}___contrast_text___orig_img"
        elif setup == "inverse_text":
            higher = f"{m}___contrast_text___contrast_img"
            lower = f"{m}___orig_text___contrast_img"
        elif setup == "forward_image":
            higher = f"{m}___orig_text___orig_img"
            lower = f"{m}___orig_text___contrast_img"
        elif setup == "inverse_image":
            higher = f"{m}___contrast_text___contrast_img"
            lower = f"{m}___contrast_text___orig_img"
        #try:
        # Sort by the columns to ensure a deterministic order
        df_sorted = df.sort_values([id_col, "prompt", "img_gen_model"]).copy()

        # Drop any existing {m}_compared column from the original DataFrame (computed for original data)
        df_sorted = df_sorted.drop(columns=[f"{m}_compared"], errors='ignore').copy()
        df_sorted.reset_index(drop=True, inplace=True)

        # For text based, we test if the highest matching text-img pair is higher than the contrast_text pair with the SAME image
        if "text" in setup:
            df_sorted[higher] = pd.to_numeric(df_sorted[higher], errors="coerce")
            df_sorted[lower] = pd.to_numeric(df_sorted[lower], errors="coerce")
            df_sorted[f"{m}_compared"] = df_sorted[higher] > df_sorted[lower]  # pre compute the comparison for all rows and afterwards filter where the highest "higher" value is in each text-image group (as we have 5 images per prompt)
            grouper = [id_col, "prompt", "img_gen_model"]
            
            df_sorted = (
                df_sorted.sort_values(by=higher, ascending=False)
                .dropna(subset=[higher])
                .drop_duplicates(grouper, keep="first")
                .reset_index(drop=True)
                .copy()
            )

        # For image based, we test if the highest matching text-img pair is higher than the contrast_img pair for all images in the group
        elif "image" in setup:
            # Compute max values per group efficiently
            max_values = (
                df_sorted.groupby([id_col, "prompt", "img_gen_model"], as_index=False)[[higher, lower]]
                .max()
            )

            max_values[f"{m}_compared"] = max_values[higher] > max_values[lower] 
            
            merged_df = df_sorted.merge(
                max_values, 
                on=[id_col, "prompt", "img_gen_model"], 
                how="left", 
                suffixes=("", "_max")
            )
            
            # Only keep rows where the original 'higher' equals the computed max 'higher'
            merged_df = merged_df[merged_df[higher] == merged_df[f"{higher}_max"]].copy()
            merged_df.drop(columns=[f"{higher}_max", f"{lower}_max"], inplace=True)
            
            # drop duplicate ID, prompt, img_gen_model combinations. It is deterministic because of the sorting. 
            # This merging is necessary, because the max compariso is only based on the metric columns in the current iteration
            merged_df = merged_df.drop_duplicates(subset=[id_col, "prompt", "img_gen_model"])
            
            # Save the result for this metric
            df_sorted = merged_df


        #except Exception as e:
        #    print(f"Error: {e}")

        try:
            #print(f"Sorted DataFrame for metric {m} and setup {setup}:")
            #print(df_sorted.columns)
            stats[m] = df_sorted[f"{m}_compared"].mean() # The mean of the comparison column is the accuracy/success rate of each metric
        except Exception as e:
            print(f"Error: {e}")
    return stats

def analyze(df, setup, id_col="prompt_id"):
    heatmap_dict = {}
    print("Processing DataFrame with shape:", df.shape)
    for mode in df["mode"].unique():
        print("Processing mode:", mode, df)
        subset = df[df["mode"] == mode]
        print("Subset shape:", subset.shape)
        heatmap_dict[mode] = print_setup_stats(subset, setup, id_col=id_col)
    return heatmap_dict

def stats_dict_to_df(stats_dict):
    """Convert a nested dictionary to a DataFrame."""
    return pd.DataFrame(stats_dict).T

def create_two_tables(df_a, df_b, title_a, title_b):
    """Produce LaTeX tables for two DataFrames."""
    save_and_print_latex_table(
        df_a,
        f"table_{title_a.replace(' ', '_')}.tex",
        caption=title_a,
        label=f"tab:{title_a.replace(' ', '_')}"
    )
    save_and_print_latex_table(
        df_b,
        f"table_{title_b.replace(' ', '_')}.tex",
        caption=title_b,
        label=f"tab:{title_b.replace(' ', '_')}"
    )

def create_four_tables(df_a, df_b, df_c, df_d, title_a, title_b, title_c, title_d):
    """Produce LaTeX tables for four DataFrames."""
    save_and_print_latex_table(
        df_a,
        f"table_{title_a.replace(' ', '_')}.tex",
        caption=title_a,
        label=f"tab:{title_a.replace(' ', '_')}"
    )
    save_and_print_latex_table(
        df_b,
        f"table_{title_b.replace(' ', '_')}.tex",
        caption=title_b,
        label=f"tab:{title_b.replace(' ', '_')}"
    )
    save_and_print_latex_table(
        df_c,
        f"table_{title_c.replace(' ', '_')}.tex",
        caption=title_c,
        label=f"tab:{title_c.replace(' ', '_')}"
    )
    save_and_print_latex_table(
        df_d,
        f"table_{title_d.replace(' ', '_')}.tex",
        caption=title_d,
        label=f"tab:{title_d.replace(' ', '_')}"
    )


def load_and_process(df, direction="forward_text", img_gen_model_filter=None, id_col="prompt_id"):
    if img_gen_model_filter:
        df = df[df["img_gen_model"] == img_gen_model_filter]

    hm = analyze(df, direction, id_col=id_col)
    df_hm = stats_dict_to_df(hm)
    return df_hm

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate and generate LaTeX tables for metrics.")
    parser.add_argument("--input", type=str, default="outputs/croc_syn/metrics/scores", help="Directory containing the metric score TSV files / OR a single TSV file to process")
    parser.add_argument("--output_file", type=str, default="outputs/croc_syn/plots/score_table.tex", help="File to save LaTeX tables")
    parser.add_argument("--save_concats", action="store_true", help="Whether to save the concatenated DataFrames as TSV files")
    parser.add_argument("--num_parts", type=int, default=10, help="Number of parts to combine for each model")
    parser.add_argument("--img_gen_model_filter", type=str, default=None, help="Optional filter for image generation model (e.g., 'flux' or 'Stable Diffusion')")
    parser.add_argument("--id_col", type=str, default="prompt_id", help="Column name for the unique identifier (default: 'prompt_id')")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    
    # Define the two text gen models (used in the filename)
    text_models = ["deepseek_r1_distill_qwen_14b", "qwen_qwq_32b"]
    text_models_short = ["ds", "qwen"]

    img_models_short = ["flux", "sd"]

    settings = ["forward_text", "inverse_text", "forward_image", "inverse_image"]

    print("Evaluating with:", args)
    
    if os.path.isdir(args.input): # Load all the sharded Score files
        all_dfs = []
        for img_model in img_models_short:
            print(f"Processing image generation model: {img_model}")
            for text_model in text_models_short:
                full_paths = [os.path.join(args.input, f"{img_model}_{text_model}", f"part{part}_scores.tsv") for part in range(args.num_parts)]
                df_list = []
                for path in full_paths:
                    if os.path.exists(path):
                        df_list.append(pd.read_csv(path, sep="\t"))
                    else:
                        print(f"Warning: File {path} does not exist and will be skipped.")
                if df_list:
                    combined_df = pd.concat(df_list, ignore_index=True)
                    combined_df["img_gen_model"] = img_model
                    combined_df["text_gen_model"] = text_model
                    combined_df["mode"] = combined_df[args.id_col].str.split("|||").str[0] # extract the mode from the prompt_id for grouping
                    all_dfs.append(combined_df)
                else:
                    print(f"Error: No valid files found for {img_model}_{text_model}.")

        df = pd.concat(all_dfs, ignore_index=True)
        if args.save_concats:
            concat_output_path = os.path.join(args.input, "concatenated_scores.tsv")
            df.to_csv(concat_output_path, sep="\t", index=False)
            print(f"Concatenated DataFrame saved to {concat_output_path}")
    else:
        df = pd.read_csv(args.input, sep="\t")

    df[args.id_col] = df[args.id_col].str.split("_____").str[0] # For grouping, we don't need the image identifier
    
    print(df.iloc[0])

    df_hm_dict = {}
    combine_dfs_list = {}
    for setting in tqdm.tqdm(settings):
        df_hm_dict[(setting, text_models_short[0])] = load_and_process(df[df["text_gen_model"] == text_models_short[0]], direction=setting, img_gen_model_filter=args.img_gen_model_filter, id_col=args.id_col)
        df_hm_dict[(setting, text_models_short[1])] = load_and_process(df[df["text_gen_model"] == text_models_short[1]], direction=setting, img_gen_model_filter=args.img_gen_model_filter, id_col=args.id_col)
        combine_dfs_list[setting] = combine_dfs(df_hm_dict[(setting, text_models_short[0])], df_hm_dict[(setting, text_models_short[1])], mode=0 if "text" in setting else 1)

    print(df_hm_dict)

    # Get all metric scores and correlate the ranking between the text generation models
    model1_values = []
    for df in [df_hm_dict[(setting, text_models_short[0])] for setting in settings]:
        for col in df.columns:
            model1_values+= df[col].tolist()
    model2_values = []
    for df in [df_hm_dict[(setting, text_models_short[1])] for setting in settings]:
        for col in df.columns:
            model2_values+= df[col].tolist()
    tau, p_value = kendalltau(model1_values, model2_values)
    print(f"Kendall Tau correlation between {text_models_short[0]} and {text_models_short[1]}: {tau:.4f} (p-value: {p_value:.4f})")


    # Mostly generated method to format the current results into a nice Latex table
    create_large_combined_table(
        combine_dfs_list[settings[0]],
        combine_dfs_list[settings[1]],
        combine_dfs_list[settings[2]],
        combine_dfs_list[settings[3]],
        title_a=f"Metrics ({settings[0]})",
        title_b=f"Metrics ({settings[1]})",
        title_c=f"Metrics ({settings[2]})",
        title_d=f"Metrics ({settings[3]})",
        output_file=args.output_file,
        caption="Combined Metrics for Evaluation t2i",
        label="tab:combined_t2i"
    )