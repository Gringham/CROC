import argparse
import torch
from PIL import Image
import tqdm
from transformers import AutoProcessor, AutoModelForCausalLM

from metrics.BaseMetric import BaseMetric

class CROCScore(BaseMetric):
    def __init__(
        self,
        model_name: str = "nllg/crocscore",
        device: str = "cuda",
        use_flash_attention: bool = True,
        base_model_name_or_path: str = "microsoft/Phi-4-multimodal-instruct",
        batch_size: int = 8,
        question = "Does this image show the following content:'{prompt}'? Answer with Yes or No.",
    ):
        """
        Loads the processor from the original base model and the fine-tuned model from a HF checkpoint directory.
        Supports batching of (prompt, image) pairs.
        """
        super().__init__()
        self.device = device
        self.model_name = model_name
        self.use_flash_attention = use_flash_attention
        self.base_model_name_or_path = base_model_name_or_path
        self.batch_size = batch_size

        self.positive_answer = "Yes"
        self.negative_answer = "No"
        self.question = question

        # 1) Processor: load from the original base model
        self.processor = AutoProcessor.from_pretrained(
            self.base_model_name_or_path,
            trust_remote_code=True,
            dynamic_hd=36,
        )

        # 2) Load the fully fine-tuned model directly from model_name
        dtype = torch.bfloat16 if self.use_flash_attention else torch.float32
        attn_impl = 'flash_attention_2' if self.use_flash_attention else 'sdpa'
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=dtype,
            trust_remote_code=True,
            _attn_implementation=attn_impl,
        ).to(self.device)
        self.model.eval()
        print(f"Loaded CROCScore model from {self.model_name} onto {self.device}. {self.get_state()}")

    def get_state(self) -> str:
        return (
            f"CROCScore using model @ {self.model_name} "
            f"(flash_attention={self.use_flash_attention}), batch_size={self.batch_size}"
        )

    def __call__(self, captions, image_paths, direct_img=False):
        # Single example: delegate to batch for simplicity
        if isinstance(captions, str):
            return self._score_one(captions, image_paths, direct_img=direct_img)

        # Batched scoring
        scores = []
        total = len(captions)
        for i in tqdm.tqdm(range(0, total, self.batch_size)):
            batch_caps = captions[i : i + self.batch_size]
            batch_paths = image_paths[i : i + self.batch_size]
            batch_scores = self._score_batch(batch_caps, batch_paths, direct_img)
            scores.extend(batch_scores)
        return scores

    def _score_one(self, prompt: str, path: str, direct_img=False) -> float:
        return self._score_batch([prompt], [path], direct_img=direct_img)[0]

    def _score_batch(self, prompts: list[str], paths: list[str], direct_img=False) -> list[float]:
        # 1) Load all images
        if not direct_img:
            imgs = [Image.open(p) for p in paths]
        else:
            imgs = paths
            
        imgs = [img.convert("RGB").resize((1024, 1024), Image.LANCZOS) for img in imgs]

        # 2) Build per-sample chat inputs
        chat_texts = []
        for prompt in prompts:
            question = self.question.format(prompt=prompt)
            single = self.processor.tokenizer.apply_chat_template(
                [{"role": "user", "content": f"<|image_1|> {question}"}],
                tokenize=False,
                add_generation_prompt=True
            )
            chat_texts.append(single)

        # 3) Tokenize and encode together with padding
        enc = self.processor(
            text=chat_texts,
            images=imgs,
            return_tensors="pt",
            padding=True,
        )
        # Move tensors to device
        for k, v in enc.items():
            if isinstance(v, torch.Tensor):
                enc[k] = v.to(self.device)

        # 4) Forward pass
        with torch.no_grad():
            outputs = self.model(
                input_ids=enc.get("input_ids"),
                attention_mask=enc.get("attention_mask"),
                input_image_embeds=enc.get("input_image_embeds"),
                image_attention_mask=enc.get("image_attention_mask"),
                image_sizes=enc.get("image_sizes"),
                input_mode=1,
            )
        # logits shape: (batch_size, seq_len, vocab_size)
        last_logits = outputs.logits[:, -1, :]  # (batch_size, vocab_size)

        # 5) Compute Yes vs. No scores per sample
        tok = self.processor.tokenizer
        yes_id = tok.convert_tokens_to_ids(self.positive_answer)
        no_id = tok.convert_tokens_to_ids(self.negative_answer)
        
        probs = torch.softmax(last_logits, dim=-1)
        p_yes = probs[:, yes_id]
        p_no = probs[:, no_id]
        # Score is difference between P(Yes) and P(No)
        return (p_yes - p_no).tolist()

if __name__ == "__main__":
    metric = CROCScore(
        model_name=f"CROCScore",
        device="cuda",
        use_flash_attention=True,
        batch_size=4,  
    )


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

    scores = metric(captions, paths)
    print(scores)

