import tqdm, os

from metrics.BaseMetric import BaseMetric
from metrics.third_party.T2IScoreScore.src.correlation_scores.run_simscore import (
    ALIGNScore,
)


class Alignscore(BaseMetric):
    def __init__(self, type="ALIGN",device=None):
        print("Loading original implementation of AlignScore")
        super().__init__()
        if type == "ALIGN":
            self.scorer = ALIGNScore()
        self.scorer.model = self.scorer.model.to("cuda")
        self.batch_size = 64

    def __call__(self, caption, path, direct_img=None):
        if type(caption) == str:
            return self.scorer.calculate_score(path, caption, direct_img=direct_img)

        elif type(caption) == list or type(caption) == tuple:
            scores = [self(c, p, direct_img) for c, p in tqdm.tqdm(zip(caption, path))]
            return scores

    def get_state(self):
        return f"https://github.com/Yushi-Hu/tifa, September 2024"


if __name__ == "__main__":
    # Instantiate ALIGN

    paths = [
        "metrics/test_images/body_parts/prompt/body_parts_1___prompt___stable_diffusion_3_5_large_turbo___image4.png",
        "metrics/test_images/body_parts/prompt/body_parts_1___prompt___gpt4___image1.png",
        "metrics/test_images/body_parts/contrast/body_parts_1___contrast___gpt4___image1.png",
        "metrics/test_images/body_parts/contrast/body_parts_1___contrast___gpt4___image4.png",
    ]

    captions = [
        "A hand with only its index finger colored red",
        "A hand with only its index finger colored red",
        "A hand with only its index finger colored red",
        "A hand with only its index finger colored red",
    ]


    paths = [
        "metrics/test_images/cat1.bmp",
        "metrics/test_images/cat2.bmp",
        "metrics/test_images/chair.bmp",
        "metrics/test_images/combined.bmp",
        "metrics/test_images/tiger.bmp",
        "metrics/test_images/cat1.bmp",
        "metrics/test_images/cat2.bmp",
        "metrics/test_images/chair.bmp",
        "metrics/test_images/combined.bmp",
        "metrics/test_images/tiger.bmp",
        "images/action/prompt/action_1___prompt___FLUX_1_schnell___image1.png",
        "images/action/prompt/action_1___prompt___FLUX_1_schnell___image1.png"
    ]
    captions = [
        "A cute cat",
        "Another cute cat",
        "A stylish chair",
        "A combined image of a cat, a tiger, and a chair",
        "A majestic tiger",
        "An ugly dog",
        "A rusty old car",
        "A dilapidated building",
        "An empty white room",
        "A boring piece of paper",
        "A soccer ball bounces.",
        "A soccer ball sits."
    ]

    paths = [
        "images/body_parts/prompt/body_parts_50___prompt___stable_diffusion_3_5_large_turbo___image19.png",
        "images/body_parts/prompt/body_parts_50___prompt___stable_diffusion_3_5_large_turbo___image19.png",
        "images/negation/alt_contrast/negation_50___alt_contrast___stable_diffusion_3_5_large_turbo___image9.png",
        "images/negation/alt_contrast/negation_50___alt_contrast___stable_diffusion_3_5_large_turbo___image9.png"
    ]

    captions = [
        "A hand with only its index finger colored yellow.",
        "A hand with only its middle finger colored yellow.",
        "An alien creature and a crowd.",
        "An alien creature and no crowd."
    ]

    metric = Alignscore()
    scores = metric(captions, paths)
    print("ALIGN", scores)
