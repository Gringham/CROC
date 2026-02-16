from vllm import LLM, SamplingParams

from project_root import CACHE_DIR

def vllm_startup_helper(model, n, vllm_sampling_mode="greedy", max_tokens=180, parallel=1, sampling_params=None):
    print("Loading vllm with: ", model, vllm_sampling_mode, max_tokens, parallel)
    
    if sampling_params == None:
        if vllm_sampling_mode == "greedy":
            sampling_params = SamplingParams(temperature=0, max_tokens=int(max_tokens))
        elif vllm_sampling_mode == "topp":
            sampling_params = SamplingParams(n, temperature=0.6, top_p=0.9, top_k=50, presence_penalty=0.5, max_tokens=int(max_tokens))
            
    if "gptq" in model.lower():
        llm = LLM(model=model, quantization="marlin", tensor_parallel_size=int(parallel), download_dir=CACHE_DIR)
    else:
        llm = LLM(model=model, tensor_parallel_size=int(parallel), download_dir=CACHE_DIR, trust_remote_code=True, max_model_len=32768, swap_space=16)# default is hanging, using ray
    return llm, sampling_params

if __name__ == "__main__":
    sampling_params = SamplingParams(temperature=0, max_tokens=180)

    # Print the sampling parameters
    print(vars(sampling_params))
    print(sampling_params._verify_greedy_sampling)
