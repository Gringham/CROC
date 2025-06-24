import argparse
import json
from pathlib import Path
import random
from torch.utils.data import DataLoader

import torch
from accelerate import Accelerator
from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
    BatchFeature,
    Trainer,
    TrainingArguments,
)

# Training script for CROCScore. Adapted from: https://huggingface.co/microsoft/Phi-4-multimodal-instruct/blob/main/sample_finetune_vision.py

_IGNORE_INDEX = -100
_TRAIN_SIZE = 16000
_EVAL_SIZE = 1000
_MAX_TRAINING_LENGTH = 8192


from datasets import load_dataset
from torch.utils.data import Dataset
from PIL import Image
from transformers import AutoProcessor
import torch

_IGNORE_INDEX = -100
_MAX_LEN = 8192

from datasets import load_from_disk
from torch.utils.data import Dataset
import base64
from io import BytesIO
from PIL import Image
import torch

_IGNORE_INDEX = -100
_MAX_LEN = 8192

class B64Data(Dataset):
    def __init__(
        self,
        local_path: str,
        split: str,
        processor,
        instruction="Does this image show {}? Answer with Yes or No.",
        max_samples=None,
        seed=42,
        save_choices_path=None,
    ):
        # 1) load full HF dataset, pick split
        full = load_from_disk(local_path)
        if split == "train":
            raw = full
        else:
            raw = full[split]

        # 2) deterministically shuffle the *row indices* and select
        all_idxs = list(range(len(raw)))
        rand = random.Random(seed)
        rand.shuffle(all_idxs)
        if max_samples is not None:
            chosen_idxs = all_idxs[:max_samples]
        else:
            chosen_idxs = all_idxs
        
        # 3) select that subset from the HF Dataset
        self.ds = raw.select(chosen_idxs)

        # 4) pre‐compute a choice-list of 0 or 1 per example
        self.choice_rng = random.Random(seed + 1)
        self.choices = [ self.choice_rng.choice([0,1]) for _ in range(len(self.ds)) ]

        # 5) optionally save the mapping & choices to disk
        if save_choices_path is not None:
            Path(save_choices_path).parent.mkdir(parents=True, exist_ok=True)
            json.dump({
                "original_indices": chosen_idxs,
                "image_choice": self.choices
            }, open(save_choices_path, "w"), indent=2)

        self.processor   = processor
        self.instruction = instruction

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        ex = self.ds[idx]
        
        # use our pre‐computed choice
        choice = self.choices[idx]
        jpg_key   = f"jpg_{choice}"
        label_key = f"label_{choice}"

        img = Image.open(BytesIO(base64.b64decode(ex[jpg_key]))).convert("RGB")

        # build prompt
        question = self.instruction.format(ex["caption"])
        user_msg = {"role": "user", "content": "<|image_1|> " + question}
        prompt = self.processor.tokenizer.apply_chat_template(
            [user_msg], tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(prompt, images=[img], return_tensors="pt")

        # answer → “Yes”/“No”
        ans_str = "Yes" if ex[label_key] == 1 else "No"
        eos = self.processor.tokenizer.eos_token or self.processor.tokenizer.sep_token
        ans_ids = self.processor.tokenizer(ans_str + eos, return_tensors="pt").input_ids

        # concat, mask, truncate as before
        input_ids = torch.cat([inputs.input_ids, ans_ids], dim=1)
        labels   = torch.full_like(input_ids, _IGNORE_INDEX)
        labels[:, -ans_ids.size(1):] = ans_ids

        if input_ids.size(1) > _MAX_LEN:
            input_ids = input_ids[:, -_MAX_LEN:]
            labels    = labels[:,    -_MAX_LEN:]

        return {
            "input_ids":          input_ids[0],
            "labels":             labels[0],
            "input_image_embeds": inputs.input_image_embeds[0],
            "image_attention_mask": inputs.image_attention_mask[0],
            "image_sizes":        inputs.image_sizes[0],
            "label":              ex[label_key],
        }



def pad_sequence(sequences, padding_side='right', padding_value=0):
    """
    Pad a list of sequences to the same length.
    sequences: list of tensors in [seq_len, *] shape
    """
    assert padding_side in ['right', 'left']
    max_size = sequences[0].size()
    trailing_dims = max_size[1:]
    max_len = max(len(seq) for seq in sequences)
    batch_size = len(sequences)
    output = sequences[0].new_full((batch_size, max_len) + trailing_dims, padding_value)
    for i, seq in enumerate(sequences):
        length = seq.size(0)
        if padding_side == 'right':
            output.data[i, :length] = seq
        else:
            output.data[i, -length:] = seq
    return output


def cat_with_pad(tensors, dim, padding_value=0):
    """
    cat along dim, while pad to max for all other dims
    """
    ndim = tensors[0].dim()
    assert all(
        t.dim() == ndim for t in tensors[1:]
    ), 'All tensors must have the same number of dimensions'

    out_size = [max(t.shape[i] for t in tensors) for i in range(ndim)]
    out_size[dim] = sum(t.shape[dim] for t in tensors)
    output = tensors[0].new_full(out_size, padding_value)

    index = 0
    for t in tensors:
        # Create a slice list where every dimension except dim is full slice
        slices = [slice(0, t.shape[d]) for d in range(ndim)]
        # Update only the concat dimension slice
        slices[dim] = slice(index, index + t.shape[dim])

        output[slices] = t
        index += t.shape[dim]

    return output


def binary_collate_fn(batch):
    """
    Training collator:
    - Right-padding so prompt+answer are at the front.
    - Creates `labels` with everything masked except the answer tokens.
    """
    # 1) text: right-pad inputs and labels
    input_ids_list = [b["input_ids"] for b in batch]
    labels_list    = [b["labels"]    for b in batch]
    input_ids = pad_sequence(input_ids_list,    padding_side="right", padding_value=0)
    labels    = pad_sequence(labels_list,       padding_side="right", padding_value=_IGNORE_INDEX)
    attention_mask = (input_ids != 0).long()

    # 2) image embeds/masks/sizes: pad-and-stack to common dims
    embeds_list = [b["input_image_embeds"]  for b in batch]
    masks_list  = [b["image_attention_mask"] for b in batch]
    sizes_list  = [b["image_sizes"]         for b in batch]
    max_crops = max(t.shape[0] for t in embeds_list)

    def pad_and_stack(tensors):
        out = []
        for t in tensors:
            pad_n = max_crops - t.shape[0]
            if pad_n > 0:
                pad_shape = (pad_n, *t.shape[1:])
                t = torch.cat([t, t.new_zeros(pad_shape)], dim=0)
            out.append(t)
        return torch.stack(out, dim=0)

    input_image_embeds   = pad_and_stack(embeds_list)
    image_attention_mask = pad_and_stack(masks_list)
    image_sizes          = pad_and_stack(sizes_list)

    return BatchFeature({
        "input_ids":             input_ids,
        "labels":                labels,
        "attention_mask":        attention_mask,
        "input_image_embeds":    input_image_embeds,
        "image_attention_mask":  image_attention_mask,
        "image_sizes":           image_sizes,
        "input_mode":            1,
    })


def binary_eval_collate_fn(batch, pad_token_id):
    """
    Evaluation collator:
    - Left-padding so that index -1 is always the model’s final real token.
    - No need for masked labels; we'll compare logits or generated text separately.
    """
    # 1) text: left-pad only inputs
    input_ids_list = [b["input_ids"] for b in batch]
    input_ids = pad_sequence(input_ids_list, padding_side="left", padding_value=pad_token_id)
    attention_mask = (input_ids != pad_token_id).long()

    # 2) image embeds/masks/sizes: same as training
    embeds_list = [b["input_image_embeds"]  for b in batch]
    masks_list  = [b["image_attention_mask"] for b in batch]
    sizes_list  = [b["image_sizes"]         for b in batch]
    max_crops = max(t.shape[0] for t in embeds_list)

    def pad_and_stack(tensors):
        out = []
        for t in tensors:
            pad_n = max_crops - t.shape[0]
            if pad_n > 0:
                pad_shape = (pad_n, *t.shape[1:])
                t = torch.cat([t, t.new_zeros(pad_shape)], dim=0)
            out.append(t)
        return torch.stack(out, dim=0)

    input_image_embeds   = pad_and_stack(embeds_list)
    image_attention_mask = pad_and_stack(masks_list)
    image_sizes          = pad_and_stack(sizes_list)

    # Also return gold labels if present in batch
    golds = [b.get("label") or b.get("answer") for b in batch]

    return golds, BatchFeature({
        "input_ids":             input_ids,
        "attention_mask":        attention_mask,
        "input_image_embeds":    input_image_embeds,
        "image_attention_mask":  image_attention_mask,
        "image_sizes":           image_sizes,
        "input_mode":            1,
    })
    



def create_model(model_name_or_path, use_flash_attention=False):
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=torch.bfloat16 if use_flash_attention else torch.float32,
        _attn_implementation='flash_attention_2' if use_flash_attention else 'sdpa',
        trust_remote_code=True,
    ).to('cuda')
    # remove parameters irrelevant to vision tasks
    del model.model.embed_tokens_extend.audio_embed  # remove audio encoder
    for layer in model.model.layers:
        # remove audio lora
        del layer.mlp.down_proj.lora_A.speech
        del layer.mlp.down_proj.lora_B.speech
        del layer.mlp.gate_up_proj.lora_A.speech
        del layer.mlp.gate_up_proj.lora_B.speech
        del layer.self_attn.o_proj.lora_A.speech
        del layer.self_attn.o_proj.lora_B.speech
        del layer.self_attn.qkv_proj.lora_A.speech
        del layer.self_attn.qkv_proj.lora_B.speech

    # TODO remove unused vision layers?

    return model


@torch.no_grad()
def evaluate(
    model,
    processor,
    eval_dataset,
    collate_fn,
    save_path=None,
    disable_tqdm=False,
    eval_batch_size=1,
    device="cuda"
):
    """
    Generic evaluation loop. If `collate_fn` returns (golds, BatchFeature),
    uses last-token logits to pick Yes/No. Computes and prints accuracy.
    """
    model.eval()
    # get token IDs for "Yes" / "No"
    yes_id = processor.tokenizer.convert_tokens_to_ids("Yes")
    no_id  = processor.tokenizer.convert_tokens_to_ids("No")

    loader = DataLoader(
        eval_dataset,
        batch_size=eval_batch_size,
        shuffle=False,
        collate_fn=lambda b: collate_fn(b, processor.tokenizer.pad_token_id)
                     if collate_fn is binary_eval_collate_fn
                     else collate_fn(b)
    )

    all_preds, all_labels = [], []

    for batch in tqdm(loader, disable=disable_tqdm, desc="Evaluating"):
        if isinstance(batch, tuple) and len(batch) == 2:
            golds, inputs = batch
            # normalize golds to 0/1
            labels = [1 if str(g).strip().lower() == "yes" else 0 for g in golds]
        else:
            inputs = batch
            # if using masked labels, get them from inputs["labels"]
            token_labels = inputs["labels"][:, -1]
            labels = (token_labels == yes_id).long().tolist()

        # move all tensor fields to device
        for k,v in inputs.items():
            if isinstance(v, torch.Tensor):
                inputs[k] = v.to(device)

        # ensure image embeds have 5 dims [B, N, C, H, W]
        if inputs["input_image_embeds"].ndim == 4:
            inputs["input_image_embeds"]   = inputs["input_image_embeds"].unsqueeze(1)
            inputs["image_attention_mask"] = inputs["image_attention_mask"].unsqueeze(1)

        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            input_image_embeds=inputs["input_image_embeds"],
            image_attention_mask=inputs["image_attention_mask"],
            image_sizes=inputs["image_sizes"],
            input_mode=inputs.get("input_mode", 1),
        )

        # pick the logits for the final real token
        last_logits = outputs.logits[:, -1, :]
        yes_logits, no_logits = last_logits[:, yes_id], last_logits[:, no_id]
        preds = (yes_logits > no_logits).long().tolist()

        all_preds.extend(preds)
        all_labels.extend(labels)

    # compute accuracy
    total   = len(all_labels)
    correct = sum(p == l for p, l in zip(all_preds, all_labels))
    acc = correct / total
    print(f"[Eval] acc = {acc*100:.2f}% ({correct}/{total})")

    # optionally save detailed results
    if save_path:
        with open(save_path, "w") as f:
            json.dump({
                "predictions": all_preds,
                "labels":      all_labels,
                "accuracy":    acc
            }, f, indent=2)

    return acc

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model_name_or_path',
        type=str,
        default='microsoft/Phi-4-multimodal-instruct',
        help='Model name or path to load from',
    )
    parser.add_argument('--use_flash_attention', action='store_true', help='Use Flash Attention')
    parser.add_argument('--output_dir', type=str, default='./outputs_repr/', help='Output directory')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument(
        '--batch_size_per_gpu',
        type=int,
        default=1,
        help='Batch size per GPU (adjust this to fit in GPU memory)',
    )
    parser.add_argument(
        '--dynamic_hd',
        type=int,
        default=36,
        help='Number of maximum image crops',
    )
    parser.add_argument(
        '--num_train_epochs', type=int, default=1, help='Number of training epochs'
    )
    parser.add_argument('--learning_rate', type=float, default=5.0e-6, help='Learning rate')
    parser.add_argument('--wd', type=float, default=0.01, help='Weight decay')
    parser.add_argument('--no_tqdm', dest='tqdm', action='store_false', help='Disable tqdm')
    parser.add_argument('--full_run', action='store_true', help='Run the full training and eval')
    parser.add_argument('--resume_from_checkpoint', action='store_true', help='Resume from checkpoint')
    args = parser.parse_args()

    print(args)

    accelerator = Accelerator()

    with accelerator.local_main_process_first():
        processor = AutoProcessor.from_pretrained(
            args.model_name_or_path,
            trust_remote_code=True,
            dynamic_hd=args.dynamic_hd,
        )
        model = create_model(
            args.model_name_or_path,
            use_flash_attention=args.use_flash_attention,
        )
        
    train_dataset = B64Data(
        'CROC/metrics/custom_pickscore/clean_train',
        split='train',
        processor=processor,
        max_samples=_TRAIN_SIZE,
        save_choices_path=args.output_dir + '/train_choicesTS.json',
    )
    eval_dataset = B64Data(
        'CROC/metrics/custom_pickscore/final_dataset',
        split='dev',
        processor=processor,
        max_samples=_EVAL_SIZE,
    ) 
    
    #print(train_dataset.ds[0])
    #print(train_dataset[0])
        
    # tune vision encoder and lora
    model.set_lora_adapter('vision')
    for param in model.model.embed_tokens_extend.image_embed.parameters():
        param.requires_grad = True    
        
    # TESTING
    for block in model.model.layers[-2:]:
        for p in block.parameters():
            p.requires_grad = True

    # 4) (Optional) Un-freeze the language head so you can re-learn the final projection:
    for p in model.lm_head.parameters():
        p.requires_grad = True
    
    num_gpus = accelerator.num_processes
    print(f'training on {num_gpus} GPUs')
    assert (
        args.batch_size % (num_gpus * args.batch_size_per_gpu) == 0
    ), 'Batch size must be divisible by the number of GPUs'
    gradient_accumulation_steps = args.batch_size // (num_gpus * args.batch_size_per_gpu)

    if args.use_flash_attention:
        fp16 = False
        bf16 = True
    else:
        fp16 = True
        bf16 = False

    # hard coded training args
    training_args = TrainingArguments(
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.batch_size_per_gpu,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={'use_reentrant': False},
        gradient_accumulation_steps=gradient_accumulation_steps,
        optim='adamw_torch',
        adam_beta1=0.9,
        adam_beta2=0.95,
        adam_epsilon=1e-7,
        learning_rate=args.learning_rate,
        weight_decay=args.wd,
        max_grad_norm=1.0,
        lr_scheduler_type='linear',
        warmup_steps=100,
        logging_steps=10,
        output_dir=args.output_dir,
        save_strategy='steps',
        save_steps=1000,
        save_total_limit=10,
        bf16=bf16,
        fp16=fp16,
        resume_from_checkpoint=args.resume_from_checkpoint,
        remove_unused_columns=False,
        report_to='none',
        deepspeed=None,
        disable_tqdm=not args.tqdm,
        dataloader_num_workers=4,
        ddp_find_unused_parameters=True,  # for unused SigLIP layers
    )

    # eval before fine-tuning
    out_path = Path(training_args.output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    #acc = evaluate(
    #    model,
    #    processor,
    #    eval_dataset,
    #    collate_fn=binary_eval_collate_fn,
    #    save_path=out_path/"eval.json",
    #    disable_tqdm=not args.tqdm,
    #    eval_batch_size=args.batch_size_per_gpu,
    #    )
    #if accelerator.is_main_process:
    #    print(f'Accuracy before finetuning: {acc}')

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=binary_collate_fn,
        train_dataset=train_dataset,
    )
    trainer.train()
    trainer.save_model()
    accelerator.wait_for_everyone()

    # eval after fine-tuning (load saved checkpoint)
    # first try to clear GPU memory
    del model
    del trainer
    __import__('gc').collect()
    torch.cuda.empty_cache()

    # reload the model for inference
    model = AutoModelForCausalLM.from_pretrained(
        training_args.output_dir,
        torch_dtype=torch.bfloat16 if args.use_flash_attention else torch.float32,
        trust_remote_code=True,
        _attn_implementation='flash_attention_2' if args.use_flash_attention else 'sdpa',
    ).to('cuda')

    acc = acc = evaluate(
            model,
            processor,
            eval_dataset,
            collate_fn=binary_eval_collate_fn,
            save_path=out_path/"eval.json",
            disable_tqdm=not args.tqdm,
            eval_batch_size=args.batch_size_per_gpu,
            )
    if accelerator.is_main_process:
        print(f'Accuracy after finetuning: {acc}')


if __name__ == '__main__':
    main()