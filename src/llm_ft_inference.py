import torch
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer

output_dir = "llama2_7b_hf_common_properties"

# load base LLM model and tokenizer
model = AutoPeftModelForCausalLM.from_pretrained(
    output_dir,
    low_cpu_mem_usage=True,
    torch_dtype=torch.float16,
    load_in_4bit=True,
)
tokenizer = AutoTokenizer.from_pretrained(output_dir)


prompt = f"""### Instruction:
Use the Input below to identify the common property or characteristic shared by all these concepts.

### Input:
{sample['concept_list']}

### Response:
{sample['shared_property']}
"""


input_ids = tokenizer(prompt, return_tensors="pt", truncation=True).input_ids.cuda()
# with torch.inference_mode():
outputs = model.generate(
    input_ids=input_ids, max_new_tokens=100, do_sample=True, top_p=0.9, temperature=0.9
)

print(f"Prompt:\n{sample['response']}\n")
print(
    f"Generated instruction:\n{tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True)[0][len(prompt):]}"
)
print(f"Ground truth:\n{sample['instruction']}")


from peft import AutoPeftModelForCausalLM

model = AutoPeftModelForCausalLM.from_pretrained(
    args.output_dir,
    low_cpu_mem_usage=True,
)

# Merge LoRA and base model
merged_model = model.merge_and_unload()

# Save the merged model
merged_model.save_pretrained("merged_model", safe_serialization=True)
tokenizer.save_pretrained("merged_model")

# push merged model to the hub
# merged_model.push_to_hub("user/repo")
# tokenizer.push_to_hub("user/repo")
