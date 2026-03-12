# import torch
# from transformers import AutoTokenizer, AutoModelForCausalLM

# model_name = "Qwen/Qwen3-4B-Instruct-2507"

# tok = AutoTokenizer.from_pretrained(model_name)
# # model = AutoModelForCausalLM.from_pretrained(
# #     model_name,
# #     torch_dtype="auto",
# #     device_map="auto",
# # )
# model = AutoModelForCausalLM.from_pretrained(
#     model_name,
#     dtype=torch.float16,     # reduces RAM vs float32
#     device_map="cpu",              # don't touch GPU
# )


# messages = [{"role": "user", "content": "Say hello in one short sentence."}]
# text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
# inputs = tok([text], return_tensors="pt").to(model.device)

# out = model.generate(**inputs, max_new_tokens=30, do_sample=False)
# print(tok.decode(out[0], skip_special_tokens=True))


import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "Qwen/Qwen3-4B-Instruct-2507"

tok = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    # FIXED: Use 'torch_dtype', not 'dtype'
    # FIXED: Use float32 for CPU compatibility (or bfloat16 if your CPU is new)
    torch_dtype=torch.float32, 
    device_map="cpu",
    trust_remote_code=True # Often needed for specific/new Qwen versions
)

messages = [{"role": "user", "content": "Hur mycket är 2+2?"}]
text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

# Generate inputs
inputs = tok([text], return_tensors="pt").to(model.device)

# Run generation
out = model.generate(**inputs, max_new_tokens=30, do_sample=False)
print(tok.decode(out[0], skip_special_tokens=True))