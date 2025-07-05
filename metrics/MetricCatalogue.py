import os
import tempfile
import traceback
import pandas as pd
import torch
from tqdm import tqdm

from metrics.VQAScore import VQAScore
from metrics.PickScore import PickScore
from metrics.Alignscore import Alignscore
from metrics.BVQA import BVQA
from metrics.ClipScore import ClipScore
from metrics.RandomScore import RandomScore
from CROC.crocscore import CROCScore


class MetricCatalogue:
    def __init__(self):
        self.metrics = {
            "BVQA": (BVQA, [], {}),
            "CLIPScore": (ClipScore, [], {}),
            "CLIPScore_Large": (ClipScore, ["openai/clip-vit-large-patch14"], {}),
            "SSAlign": (Alignscore, ["ALIGN"], {}),
            "PickScore": (PickScore, [], {}),
            "VQAScore": (VQAScore, [], {}),
            "Blip2ITM": (VQAScore, [], {"base": "blip2itm"}),
            "RandomScore": (RandomScore, [], {}),
            "PhiScore_R2": (CROCScore, [], {"checkpoint_dir": "./tune_phi/outputs_repr"}),
        }
        self.subset = self.metrics

    def list_metrics(self):
        return sorted(self.metrics.keys())

    def select_subset(self, subset):
        self.subset = {k: self.metrics[k] for k in subset if k in self.metrics}
        return self.subset

    def unselect_subset(self):
        self.subset = None

    def apply_subset(
        self,
        image_path_column,
        prompt_column,
        data_df,
        resume=True,
        suffix="",
        output_file=None,
        alt_prev_file=None
    ):
        """
        Apply the selected metrics on data_df. The image path column should contain all the paths to the images and the prompt column should contain all respectife prompts. 
        alt_prev_file allows to resume from a different file than the output file
        
        Resume behavior:
          * If resume=True and output_file exists, the existing results are loaded.
          * New rows (i.e. with valid image paths) that are in data_df but not in the saved file are added.
          * For each metric column (named metric+suffix) only rows with missing scores are computed.
          * Already computed scores are not re-computed.
        """
         # Use either the subset (if selected) or all metrics.
        metrics_to_apply = self.subset if self.subset is not None else self.metrics
                
        # Ensure that each metric column exists (initialize with NaN)
        for metric_name in metrics_to_apply.keys():
            col_name = metric_name + suffix
            if col_name not in data_df.columns:
                data_df[col_name] = float("nan")
                
        print(f"Applying metrics: {list(metrics_to_apply.keys())}")
        print(f"Data shape: {data_df.shape}")

        # If resuming, load existing results and merge with new data.
        if resume and ((output_file is not None and os.path.exists(output_file)) or (alt_prev_file is not None and os.path.exists(alt_prev_file))):
            if alt_prev_file is not None and os.path.exists(alt_prev_file):
                print("Resuming from alt_prev_file", alt_prev_file)
                try:
                    existing_df = pd.read_csv(alt_prev_file, sep="\t")
                except Exception as e:
                    raise Exception(f"Error reading existing output file: {e}")
            else:
                try:
                    existing_df = pd.read_csv(output_file, sep="\t")
                except Exception as e:
                    raise Exception(f"Error reading existing output file: {e}")

            for column in existing_df.columns:
                if column not in data_df.columns:
                    data_df[column] = [None] * len(data_df)
                                
            if "img_paths_prompts" in data_df:
                img_col = "img_paths_prompts"
            elif "contrast_image" in data_df:
                img_col = "contrast_image"
                            
            # Drop duplicates based on the 'img_paths_prompts' column
            initial_count = len(data_df)
            data_df.drop_duplicates(subset=[img_col], inplace=True)
            
            duplicates_dropped = initial_count - len(data_df)
            print(f"Dropped {duplicates_dropped} duplicates for {img_col}")
            
            if "img_paths_prompts" in existing_df:
                img_col2 = "img_paths_prompts"
            elif "contrast_image" in existing_df:
                img_col2 = "contrast_image"
            
            # Drop duplicates based on the 'img_paths_prompts' column
            initial_count = len(existing_df)
            existing_df.drop_duplicates(subset=[img_col2], inplace=True)
            print(f"Dropped {duplicates_dropped} duplicates for {img_col2}")

            # Merge so that every row in data_df is present; existing scores will be carried over.
            print("Data shape before merge:", data_df[img_col])

            data_df.set_index(img_col, inplace=True)
            existing_df.set_index(img_col, inplace=True)
            print("Data shape after merge:", data_df.reset_index()[img_col])

            data_df.update(existing_df)
            full_results_df = data_df.reset_index()

        elif output_file is not None and os.path.exists(output_file):
            raise Exception(f"Output file {output_file} already exists, please remove it or set resume=True")
        else:
            full_results_df = data_df.copy()

        # Set device.
        device = torch.device("cuda:0")

        # Determine which metrics are already complete (all valid rows have non-null scores).
        completed_metrics = set()
        for metric_name in metrics_to_apply.keys():
            metric_col = metric_name + suffix
            print(full_results_df)
            valid_rows = full_results_df[full_results_df[image_path_column].apply(os.path.exists)]
            print("Valid rows:", full_results_df[image_path_column][0])
            if valid_rows[metric_col].notna().all():
                completed_metrics.add(metric_name)
        if resume:
            print(f"Resuming from completed metrics: {completed_metrics}")

        # Process each metric individually.
        for metric_name, metric_info in tqdm(metrics_to_apply.items(), desc="Metrics Started", total=len(metrics_to_apply)):
            # Skip metrics that are already complete.
            if metric_name in completed_metrics:
                print(f"Metric {metric_name} already computed for all valid rows, skipping.")
                continue

            metric_col = metric_name + suffix

            # Identify the rows needing computation: valid image paths and missing metric score.
            missing_indices = []
            valid_image_paths = []
            valid_captions = []
            for idx, row in full_results_df.iterrows():
                image_path = row[image_path_column]
                if not os.path.exists(image_path):
                    continue
                # Mark for computation if the value is NaN.
                if pd.isna(row[metric_col]):
                    missing_indices.append(idx)
                    valid_image_paths.append(image_path)
                    valid_captions.append(row[prompt_column])

            if not missing_indices:
                print(f"Metric {metric_name} already computed for all valid rows, skipping.")
                continue

            # Instantiate the metric.
            metric_class, args, kwargs = metric_info
            try:
                kwargs_with_device = kwargs.copy()
                kwargs_with_device["device"] = device
                metric = metric_class(*args, **kwargs_with_device)
            except TypeError as e:
                if "unexpected keyword argument 'device'" in str(e):
                    metric = metric_class(*args, **kwargs)
                else:
                    print(f"Error initializing metric {metric_name}: {e}")
                    traceback.print_exc()
                    continue
            except Exception as e:
                print(f"Error initializing metric {metric_name}: {e}")
                traceback.print_exc()
                continue

            # Compute scores in chunks of 4k rows.
            print(f"Computing metric {metric_name} for {len(missing_indices)} rows. Sample image paths: {valid_image_paths[:3]}")
            chunk_size = 4000
            all_scores = []
            for i in range(0, len(valid_captions), chunk_size):
                print(f"Processing chunk starting at index {i} to {i+chunk_size} of {len(valid_captions)}")
                batch_captions = valid_captions[i:i+chunk_size]
                batch_image_paths = valid_image_paths[i:i+chunk_size]
                
                if metric_name == "BVQA":
                    # Create a temporary directory with a custom prefix that includes the current chunk index
                    temp_dir = tempfile.mkdtemp(prefix=f"bvqa_intermediate_{i}_")
                    metric.out_dir = temp_dir
                
                try:
                    print("Batch info, len batch:", len(batch_captions), batch_captions[-5:], batch_image_paths[-5:], metric_name)
                    batch_scores = metric(batch_captions, batch_image_paths)
                except Exception as e:
                    print(f"Error computing metric {metric_name} for chunk starting at index {i}: {e}")
                    traceback.print_exc()
                    continue
                # Update full_results_df for the corresponding indices for this chunk.
                for idx, score in zip(missing_indices[i:i+chunk_size], batch_scores):
                    full_results_df.at[idx, metric_col] = score
                # Save progress to file after each chunk.
                if output_file is not None:
                    full_results_df.to_csv(output_file, sep="\t", index=False)
                all_scores.extend(batch_scores)

            # Clean up and free GPU memory.
            del metric
            torch.cuda.empty_cache()

        # Save the final results if an output file was specified.
        if output_file is not None:
            full_results_df.to_csv(output_file, sep="\t", index=False)

        return full_results_df
