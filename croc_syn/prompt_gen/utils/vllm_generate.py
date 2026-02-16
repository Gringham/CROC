import torch
from croc_syn.prompt_gen.utils.chatTemplates import apply_chat_template
from croc_syn.prompt_gen.utils.vllm_startup_helper import vllm_startup_helper

def vllm_generate(
    prompt_ids,
    prompts,
    sampling_params=None,
    model="deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
    parallel=1,
    num_generations=5
):

    # Initialize the vLLM pipeline with the specified parameters
    llm, sampling_params = vllm_startup_helper(
        model, 
        n=num_generations, 
        max_tokens=1024, 
        vllm_sampling_mode="topp", 
        parallel=parallel
    )
    
    # Ensure prompt_ids and prompts have the same length
    if len(prompt_ids) != len(prompts):
        raise ValueError("The length of prompt_ids and prompts must be the same.")
    
    # Apply the chat template to the prompts
    input_data = apply_chat_template(model, prompts, llm, print_samples=True)
    
    # Generate outputs
    # Assuming llm.generate returns a list where each element corresponds to a prompt
    # and contains an 'outputs' attribute which is a list of generated outputs
    raw_outputs = llm.generate(input_data, sampling_params)
    
    # Initialize the results dictionary
    results = {}
    for prompt_id, output in zip(prompt_ids, raw_outputs):
        results[prompt_id] = [gen_output.text for gen_output in output.outputs]

    del llm
    torch.cuda.empty_cache()

    return results
