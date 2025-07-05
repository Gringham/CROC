This folder contains the files that were used to:

1. Put CROC_syn into the format used to train CROCScore (`hf_dataset_prep`)
2. Train CROCScore (`train.py`)
3. Apply CROCScore (in a format that can be executed with the MetricCatalogue Class) (`CROCScore.py`)

Before putting the dataset into the format, you need to prepare it by following the generation steps in croc_syn

For step 1:  
* Run `create_hf_dataset` to create a dataset with images in B64
* Run `prep_part2` to add ground truth labels based on whether the image and text are matching or not
* Run `prep_part3` to add another column that would be required for PickSCore training scripts. However, this was not used for CROCScore
* Run `adapt_dataset` to filter the full dataset for a smaller, more effective subset. This was applied for the CROCScore training data. 