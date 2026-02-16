# Metrics
This fodler contains wrapper scripts for the metrics that we reported in our paper:  
`AlignScore`, `BVQA`, `ClipScore`, `PickScore`, `VQAScore` (+ `GPT4` / +`Blip2Itm`). The respective sources are described in the comments of each metric. There is also a `RandomScore` that returns random scores for every data sample. Each metric implements a class `BaseMetric`.

Some metrics were not pip installable or required small modifications. Therefore, their source code  was added in the `third_party` folder (VQAScore, AlignScore, BVQA).

Each metric can be executed as follows (with the example of PickScore):  
```python
metric = PickScore()
    paths = ["vmen/metrics/test_images/cat1.bmp"]
    captions = ["A cute cat"]

    scores = metric(captions, paths)
```


To apply multiple metrics at once you can use the class `MetricCatalogue`. We leverage this class in `apply_bench` to compute the CROC_syn benchmarks reported in our paper.  
The setups are as follows:  
1_1 --> Text-based forward  
1_1_inverse --> Text-based inverse  
2_1 --> Image-based forward  
2_1_inverse --> Image-based inverse

subject_property --> Property Variation

For distributed execution with Slurm, please view the scripts in croc_syn. The scripts support chunking parameters.