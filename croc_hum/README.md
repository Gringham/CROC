#CROC_HUM
The CROChum dataset is available here: https://huggingface.co/datasets/nllg/croc_hum/tree/main
The respective images are packaged in a tar.gz file in the same repo: https://huggingface.co/datasets/nllg/croc_hum/blob/main/croc_hum.tar.gz

To evaluate a metric on CROC_hum, first use the slurm script `metric_apply.sh` (or `apply_core_metrics.py`) while specifying the ectracted dataset and metadata file (the metadata file should be placed on the same level as the extracted folders).

Outputs will be written to `outputs/croc_hum/metrics/scores`. Next, the script `create_heatmap.py` can be used to perform the evaluation and plot them as presented in the paper.

To add your own metric, simply go to `metrics` and inherit the `BaseMetric.py` class. Then, add your metric to the dictionary at the top of the `MetricCatalogue.py` script.
