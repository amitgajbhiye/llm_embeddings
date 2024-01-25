import torch
import pandas as pd
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

data_files = "data/cnet_chatgpt/prompts_file.tsv"
df = pd.read_csv(data_files, sep="\t", header=0)[1001:1021]

inf_prompt = f"""### Instruction:
Use the Input below to identify the common property or characteristic shared by all these concepts.

### Input:
<CONCEPT_LIST>

### Response:
"""

inf_prompts = []

for _, concept_list, true_shared_prop in df.values:
    # inf_prompts.append(prompt.replace("<CONCEPT_LIST>", concept_list))

    prompt = inf_prompt.replace("<CONCEPT_LIST>", concept_list)

    input_ids = tokenizer(prompt, return_tensors="pt", truncation=True).input_ids.cuda()
    # with torch.inference_mode():
    outputs = model.generate(
        input_ids=input_ids,
        max_new_tokens=100,
        do_sample=True,
        top_p=0.9,
        temperature=0.9,
    )

    print(f"Prompt:\n{prompt}\n")
    print(
        f"Generated Shared Property:\n{tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True)[0][len(prompt):]}"
    )
    print(f"Ground truth:\n{true_shared_prop}")
