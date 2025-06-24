from torchvision import transforms
from PIL import Image
import torch
import tqdm
from torch.utils.data import DataLoader, Dataset

from vmen.metrics.BaseMetric import BaseMetric

# Wrapper for Torchmetrics ClipScore https://lightning.ai/docs/torchmetrics/stable/multimodal/clip_score.html

# Custom dataset for handling image paths and captions
class CaptionImageDataset(Dataset):
    def __init__(self, image_paths, captions):
        self.image_paths = image_paths
        self.captions = captions

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        return self.image_paths[idx], self.captions[idx]

# Custom collate function to avoid default stacking
def custom_collate_fn(batch):
    # batch is a list of tuples (image_path, caption)
    image_paths, captions = zip(*batch)
    return list(image_paths), list(captions)

# Implementation of CLIPScore metric using torchmetrics
class ClipScore(BaseMetric):
    def __init__(self, model_name_or_path="openai/clip-vit-base-patch16"):
        from torchmetrics.multimodal.clip_score import CLIPScore as TorchCLIPScore
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Transformation: Resize the image, convert to tensor ([0,1]), then scale to [0,255]
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x * 255)
        ])
        # Instantiate the torchmetrics CLIPScore.
        self.metric = TorchCLIPScore(model_name_or_path=model_name_or_path).to(self.device)

    def img2tensor(self, image):
        """Converts a PIL image to a tensor using the defined transformation."""
        return self.transform(image)

    def process_batch(self, paths, captions, batch_size=16):
        # Create dataset and DataLoader using the custom collate function.
        dataset = CaptionImageDataset(paths, captions)
        dataloader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate_fn
        )

        results = []
        try:
            for image_batch, caption_batch in tqdm.tqdm(dataloader):
                # Open images as PIL Images.
                images = [Image.open(path).convert("RGB") for path in image_batch]
                # Transform each image and stack into a tensor.
                image_tensors = torch.stack([self.img2tensor(image) for image in images]).to(self.device)
                # Compute the metric for each image-caption pair.
                batch_results = [self.metric(image, caption).item() 
                                 for image, caption in zip(image_tensors, caption_batch)]
                results.extend(batch_results)
            return results
        except Exception as e:
            print("Error occurred during batch processing. Identifying faulty sample...")
            # Iterate over individual samples to find the faulty one.
            for idx in range(len(dataset)):
                sample = custom_collate_fn([dataset[idx]])
                image_path, caption = sample[0], sample[1]
                try:
                    # Attempt to process the individual sample.
                    image = Image.open(image_path).convert("RGB")
                    image_tensor = self.img2tensor(image).to(self.device)
                    _ = self.metric(image_tensor, caption)
                except Exception as inner_e:
                    print("Faulty sample found:")
                    print("Index:", idx)
                    print("Image path:", image_path)
                    print("Caption:", caption)
                    break
            raise e

    def __call__(self, caption, paths):
        if isinstance(caption, str):
            # Single image and caption processing.
            image = Image.open(paths).convert("RGB")
            image_tensor = self.img2tensor(image).to(self.device)
            return self.metric(image_tensor, caption).item()
        elif isinstance(caption, list):
            # Batch processing.
            return self.process_batch(paths, caption, batch_size=1024)

    def get_state(self):
        return f"CLIPScore: torchmetrics_version="

if __name__ == "__main__":
    # Instantiate the CLIPScore metric using a larger model variant.
    metric = ClipScore("openai/clip-vit-large-patch14")

    paths = [
        'vmen/metrics/test_images/cat1.bmp',
        'vmen/metrics/test_images/cat2.bmp',
        'vmen/metrics/test_images/chair.bmp',
        'vmen/metrics/test_images/combined.bmp',
        'vmen/metrics/test_images/tiger.bmp',
        'vmen/metrics/test_images/cat1.bmp',
        'vmen/metrics/test_images/cat2.bmp',
        'vmen/metrics/test_images/chair.bmp',
        'vmen/metrics/test_images/combined.bmp',
        'vmen/metrics/test_images/tiger.bmp'
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
        "A boring piece of paper"
    ]

    print("\n" + metric.get_state())
    scores = metric(captions, paths)
    print(scores)
