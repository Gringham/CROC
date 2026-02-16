from io import BytesIO
from tqdm import tqdm

from PIL import Image

from metrics.BaseMetric import BaseMetric
from metrics.third_party.t2v_metrics import t2v_metrics

class VQAScore(BaseMetric):
    def __init__(self, batch_size=8, base="standard", openai_key=None):
        super().__init__()
        self.device = "cuda"
        self.base = base
        if self.base == "standard":
            print("Using standard VQAScore model")
            self.model = t2v_metrics.VQAScore(
                model='clip-flant5-xxl',
                cache_dir="<CACHE_DIR>"
            )
        elif self.base == "gpt4o":
            print("Using GPT4O model")
            self.model = t2v_metrics.get_score_model(model="gpt-4o", device="cuda", openai_key=openai_key, top_logprobs=20)
        elif self.base == "blip2itm":
            print("Using BLIP2-ITM model")
            self.model = t2v_metrics.ITMScore(model='blip2-itm') 
            
        self.batch_size = batch_size

    def __call__(self, caption, image_path):
        if isinstance(caption, str):
            return self.model(images=[image_path], texts=[caption]).item()
        elif isinstance(caption, list):
            return [self.model(images=[img], texts=[cap]).item() for img, cap in tqdm(zip(image_path, caption))]
        elif isinstance(caption, tuple):
            # assert len(caption) == len(image_path)
            # Cast to list
            caption = list(caption)
            # if image paths are a tuple, cast to list
            if isinstance(image_path, tuple):
                image_path = list(image_path)
            assert len(caption) == len(image_path)
            return [self.model(images=[img], texts=[cap]).item() for img, cap in tqdm(zip(image_path, caption))]
        else:
            raise ValueError("Caption must be a string or a list of strings, but got {}: {}".format(type(caption), caption))

    def get_state(self):
        return f"VQAScore initialized with device={self.device} and batch_size={self.batch_size}"

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
    metric = VQAScore(batch_size=4, base="gpt4o", openai_key="<OPENAI_KEY>")

    from PIL import Image

    # Define paths to the original images and the new paths for the converted images
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
    ]

    paths = [path.replace(".bmp", ".jpg") for path in paths]
    
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

    # Combined test
    scores = metric(captions, paths)
    print(scores)
