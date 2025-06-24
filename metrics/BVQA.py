from vmen.metrics.BaseMetric import BaseMetric
from vmen.project_root import join_with_root

# Implementation adapted from https://github.com/Karine-Huang/T2I-CompBench - see third party folder

class BVQA(BaseMetric):
    def __init__(self):
        super().__init__()
        
        self.out_dir = "bvqa_intermediate"
        
    def __call__(self, caption, image_path):
        from vmen.metrics.third_party.T2I_CompBench.BLIPvqa_eval.BLIP_vqa import main

        if type(caption) == str:
            return main(image_path, caption, out_dir=self.out_dir).item()
        elif type(caption) == list:
            return main([(i, c) for i, c in zip(image_path, caption)], out_dir=self.out_dir).cpu().numpy().tolist()
    
    def get_state(self):
        return f"BVQA: T2I-CompBench  August 2024, Small modifications"
    
if __name__ == "__main__":
    # Instantiate the CLIP score metric
    metric = BVQA()

    paths = ['cat1.bmp', 
             'cat2.bmp', 
             'chair.bmp', 
             'combined.bmp', 
             'tiger.bmp',
             'cat1.bmp', 
             'cat2.bmp', 
             'chair.bmp', 
             'combined.bmp', 
             'tiger.bmp']*1000
    
    paths = [join_with_root("metrics/test_images/" + p) for p in paths]
    
    captions = ["A cute cat",
                "Another cute cat",
                "A stylish chair",
                "A combined image of a cat, a tiger, and a chair",
                "A majestic tiger",
                "An ugly dog",
                "A rusty old car",
                "A dilapidated building",
                "An empty white room",
                "A boring piece of paper"]*1000
        
    input = [(p, c) for p, c in zip(paths, captions)]


    print("\n" + metric.get_state())


    # Combined test

    scores = metric(captions, paths)
    print(scores)