This folder contains scripts to generate the contrastive prompt sets of CROC.

`1_pre_prompt_gen.py` uses `taxonomy.json` to fill templates that prompt LLMs to generate contrastive prompts.
`2_prompt_gen.py` uses these prompts to generate the actual contrastive prompts
`3_post_prompt_gen.py` extracts these prompts and saves them to a dedicated format

```
#Example Usage

python croc_syn/prompt_gen/1_pre_prompt_gen.py

# For a test run, generate 1 prompt and contrast for 20 samples
NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES="0,1,2,3" python croc_syn/prompt_gen/2_prompt_gen.py  --num_generations 1 --max_samples 20

python croc_syn/prompt_gen/3_post_prompt_gen.py 

# Outputs are saved to outputs/croc_syn/prompt_gen
```

The prompting guides in the utils folder are derived from https://stability.ai/learning-hub/stable-diffusion-3-5-prompt-guide and https://www.giz.ai/flux-1-prompt-guide/.