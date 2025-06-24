from io import BytesIO
from PIL import Image
from tqdm import tqdm  # <-- Import tqdm for progress display

# Wrappers for vqascore and blip2itm from https://github.com/linzhiqiu/t2v_metrics

from vmen.metrics.BaseMetric import BaseMetric

class VQAScore(BaseMetric):
    def __init__(self, batch_size=8, base="standard", openai_key=None):
        from vmen.metrics.third_party.t2v_metrics import t2v_metrics

        super().__init__()
        self.device = "cuda"
        self.base = base
        if self.base == "standard":
            self.model = t2v_metrics.VQAScore(
                model='clip-flant5-xxl',
                cache_dir="//.cache"
            )
        elif self.base == "gpt4o":
            self.model = t2v_metrics.get_score_model(model="gpt-4o", device="cuda", openai_key=openai_key, top_logprobs=20)
        elif self.base == "blip2itm":
            self.model = t2v_metrics.ITMScore(model='blip2-itm') 
        else:
            raise ValueError(f"Unknown base model: {self.base}")
            
        self.batch_size = batch_size

    def __call__(self, caption, image_path):
        if isinstance(caption, str):
            return self.model(images=[image_path], texts=[caption]).item()
            
        elif isinstance(caption, list):
            return [self.model(images=[img], texts=[cap]).item() for img, cap in tqdm(zip(image_path, caption))]

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
    # Instantiate the VQA score metric with a smaller batch size for demonstration.
    metric = VQAScore(batch_size=1, base="blib2-itm")
    metric = VQAScore(batch_size=1, base="gpt4o")
    metric = VQAScore(batch_size=1, base="standard")

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

    paths = [
        'vrob/vmen/metrics/test_images/cat1.bmp', 
        'vrob/vmen/metrics/test_images/cat2.bmp', 
        'vrob/vmen/metrics/test_images/chair.bmp', 
        'vrob/vmen/metrics/test_images/combined.bmp', 
        'vrob/vmen/metrics/test_images/tiger.bmp',
        'vrob/vmen/metrics/test_images/cat1.bmp', 
        'vrob/vmen/metrics/test_images/cat2.bmp', 
        'vrob/vmen/metrics/test_images/chair.bmp', 
        'vrob/vmen/metrics/test_images/combined.bmp', 
        'vrob/vmen/metrics/test_images/tiger.bmp'
    ]

    print("\n" + metric.get_state())
    

    scores = metric(captions, paths)
    print(scores)
    