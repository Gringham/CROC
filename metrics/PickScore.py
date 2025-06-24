from io import BytesIO
import tqdm
from transformers import AutoProcessor, AutoModel

from PIL import Image
import torch

from vmen.metrics.BaseMetric import BaseMetric

# Wrapper for PickScore: https://github.com/yuvalkirstain/PickScore

class PickScore(BaseMetric):
    def __init__(self):
        super().__init__()
        self.device = "cuda"
        self.processor_name_or_path = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
        self.model_pretrained_name_or_path = "yuvalkirstain/PickScore_v1"

        self.processor = AutoProcessor.from_pretrained(self.processor_name_or_path)
        self.model = (
            AutoModel.from_pretrained(self.model_pretrained_name_or_path)
            .eval()
            .to(self.device)
        )

    def __call__(self, caption, image_path):
        if type(caption) == str:
            return self.calc_probs(caption, image_path)
        elif type(caption) == list:
            return [self.calc_probs(cap, img) for img, cap in tqdm.tqdm(zip(image_path, caption))]

    def get_state(self):
        return f""

    def calc_probs(self, prompt, path):
        img = Image.open(path)
        if img.format == "BMP":
            img = self.convert_image_format(img, "JPEG")

        # preprocess
        image_inputs = self.processor(
            images=[img],
            padding=True,
            truncation=True,
            max_length=77,
            return_tensors="pt",
        ).to(self.device)

        text_inputs = self.processor(
            text=prompt,
            padding=True,
            truncation=True,
            max_length=77,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            # embed
            image_embs = self.model.get_image_features(**image_inputs)
            image_embs = image_embs / torch.norm(image_embs, dim=-1, keepdim=True)

            text_embs = self.model.get_text_features(**text_inputs)
            text_embs = text_embs / torch.norm(text_embs, dim=-1, keepdim=True)

            # score
            scores = self.model.logit_scale.exp() * (text_embs @ image_embs.T)[0]

            # get probabilities if you have multiple images to choose from
            #,probs = torch.softmax(scores, dim=-1)

        return scores.cpu().tolist()[0]

    def convert_image_format(self, image: Image.Image, output_format: str = "JPEG"):
        """
        Convert a PIL image to the specified format (JPEG or PNG) and return the image data as bytes.

        Parameters:
        - image (PIL.Image.Image): The input image to convert.
        - output_format (str): The desired output format ('JPEG' or 'PNG').

        Returns:
        - bytes: The converted image data in bytes.
        """
        output_buffer = BytesIO()
        image.save(output_buffer, format=output_format)
        output_buffer.seek(0)
        converted_image = Image.open(output_buffer)
        return converted_image


if __name__ == "__main__":
    # Instantiate the CLIP score metric
    metric = PickScore()

    paths = [
        "vmen/metrics/test_images/cat1.bmp",
        "vmen/metrics/test_images/cat2.bmp",
        "vmen/metrics/test_images/chair.bmp",
        "vmen/metrics/test_images/combined.bmp",
        "vmen/metrics/test_images/tiger.bmp",
        "vmen/metrics/test_images/cat1.bmp",
        "vmen/metrics/test_images/cat2.bmp",
        "vmen/metrics/test_images/chair.bmp",
        "vmen/metrics/test_images/combined.bmp",
        "vmen/metrics/test_images/tiger.bmp",
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
    ]

    print("\n" + metric.get_state())

    scores = metric(captions, paths)
    print(scores)
