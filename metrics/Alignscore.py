import tqdm, os
from sklearn.preprocessing import MinMaxScaler


from vmen.metrics.BaseMetric import BaseMetric
from vmen.project_root import CACHE_DIR, join_with_root

# Implementation adapted from https://github.com/michaelsaxon/T2IScoreScore (January) - see third party folder


class Alignscore(BaseMetric):
    def __init__(self, type = "ALIGN"):
        from vmen.metrics.third_party.T2IScoreScore.src.correlation_scores.run_simscore import ALIGNScore, BlipScorer, CLIPScorer


        super().__init__()
        if type == "CLIP":
            self.scorer = CLIPScorer()
        elif type == "BLIP":
            self.scorer = BlipScorer()
        elif type == "ALIGN":
            self.scorer = ALIGNScore()
        self.scorer.model = self.scorer.model.to("cuda")

    def normalize_scores(self, scores):
        # Unused method, because scaling like this throws of distributed applied scores
        scaler = MinMaxScaler()
        normalized_scores = scaler.fit_transform([[score] for score in scores])
        normalized_scores = [score[0] for score in normalized_scores]
        return normalized_scores

        
    def __call__(self, prompt, image_path):
        if type(prompt) == str:
            return self.scorer.calculate_score(image_path, prompt)
        
        elif type(prompt) == list:
            scores = [self(c, p) for c, p in tqdm.tqdm(zip(prompt, image_path))]
            return scores
    
    def get_state(self):
        return f"2024"
    
if __name__ == "__main__":
    os.environ["HF_HOME"] = CACHE_DIR

    metric = Alignscore()

    paths = ['cat1.bmp', 
             'cat2.bmp', 
             'chair.bmp', 
             'combined.bmp', 
             'tiger.bmp',
             'cat1.bmp', 
             'cat2.bmp', 
             'chair.bmp', 
             'combined.bmp', 
             'tiger.bmp']
    
    paths = [join_with_root("metrics/test_images/" + p) for p in paths]
    
    prompts = ["A cute cat",
                "Another cute cat",
                "A stylish chair",
                "A combined image of a cat, a tiger, and a chair",
                "A majestic tiger",
                "An ugly dog",
                "A rusty old car",
                "A dilapidated building",
                "An empty white room",
                "A boring piece of paper"]

    scores = metric(prompts, paths)
    print(scores)
    
