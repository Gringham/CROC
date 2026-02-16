def trim_string_by_token_length_hf(input_string, tokenizer):
    # If length is okay return the input string
    if len(tokenizer.tokenize(input_string)) <= tokenizer.model_max_length - 200:
        return input_string

    source_part = input_string.split("Source Text:")[1]

    # Tokenize the input string
    tokens = tokenizer.tokenize(source_part)
    
    # Filter out tokens that are longer than max_length
    trimmed_tokens = tokens[-(tokenizer.model_max_length - 200):]
    
    # Decode the filtered tokens back into a string
    trimmed_string = tokenizer.convert_tokens_to_string(trimmed_tokens)
    
    print(tokenizer.model_max_length, trimmed_string, input_string.replace(source_part, trimmed_string))
    print("\n\n|||")
    return input_string.replace(source_part, trimmed_string)

def apply_chat_template(model_name: str, prompts, llm=None, print_samples=False):
    print("Model name: ", model_name)

    if "openorca" in model_name.lower() or "nous-hermes" in model_name.lower():
        text = "### Instruction:\n\n{prompt}\n### Response:\n"
        res = [text.format(prompt=prompt) for prompt in prompts]
    elif "platypus2-70b" in model_name.lower():
        text = "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{prompt}\n\n### Response:\n"
        res = [text.format(prompt=prompt) for prompt in prompts]
    else:
        print("using default template")
        res = [llm.llm_engine.tokenizer.tokenizer.apply_chat_template([{"role": "user", "content": prompt}],
                                                                tokenize=False, add_generation_prompt=True)
         for prompt in prompts]

    if print_samples:
        print("Printing 3 first inputs:")
        for i in range(3):
            print(f"Prompt:\n{res[i]}\n")


    return res
