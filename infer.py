import os
import warnings
warnings.filterwarnings(
    "ignore", 
    message=r"The following generation flags are not valid and may be ignored: \['temperature', 'top_p'\]"
)
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

import torch
from tqdm import tqdm
import time
from datetime import datetime
import pandas as pd
import re
from transformers import AutoTokenizer, AutoModelForCausalLM

# === Load Data ===
df = pd.read_csv("path_to_dataset")
df["year_month"] = pd.to_datetime(df["year_month"], format="%Y-%m-%d")
df_2023 = df[df["year_month"].dt.year == 2023]

# === Settings ===
model_id = "Model_Name"
hf_token = "HuggingFace_Token"
device = torch.device("cuda:0")
dtype = torch.bfloat16
save_path = "save_path"
save_every = 25
batch_size = 8   # number of queries per batch

# === Load Model ===
tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map={"": 0},
    torch_dtype=dtype,
    token=hf_token,
)
model.eval()
if hasattr(torch, "compile"):
    model = torch.compile(model)

# === Prompt templates ===
SYSTEM_PROMPT = (
    "You are an air pollution assistant. "
    "Strictly respond to queries with a single real number only. "
    "Do not include any units, explanation, or punctuation. Just a single number."
)
USER_TEMPLATE = (
    "What is the average PM2.5 concentration (in μg/m³) in {location_name} during {month}, {year}? "
    "Give a single number only. Additional context: Aerosol Optical Depth = {aod}"
)

# === Resume logic ===
rows = []
if os.path.exists(save_path):
    existing_df = pd.read_csv(save_path)
    print(f"Resuming from saved file with {len(existing_df)} rows.")
    processed_keys = set(zip(existing_df["location_name"], existing_df["month"]))
    rows = existing_df.to_dict("records")
else:
    print("No previous file found. Starting fresh.")
    processed_keys = set()

# === Group cities ===
grouped = df_2023.groupby(["location_name", df_2023["year_month"].dt.month])
counter = len(rows)
AOD = [0, 0.2, 0.4, 0.6, 0.8, 1]

# === Batch Query Function ===
def batch_query_llm(batch_prompts):
    """Takes a list of prompt strings and returns list of float predictions."""
    inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True).to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=10,
            temperature=0,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    results = []
    # find the true prompt lengths (not including padding)
    prompt_lengths = inputs["attention_mask"].sum(dim=1).tolist()

    for i, output in enumerate(outputs):
        gen_tokens = output[prompt_lengths[i]:]   # take only generated part
        decoded = tokenizer.decode(gen_tokens, skip_special_tokens=True)
        match = re.search(r"\d+(\.\d+)?", decoded)
        results.append(float(match.group()) if match else float("nan"))
    return results

# === Main loop with batching ===
for aod in AOD:
    # collect prompts and metadata
    batch_prompts, meta_info = [], []
    for (location_name, month_num), _ in tqdm(grouped, desc=f"AOD {aod}"):
        month_name = datetime(1900, month_num, 1).strftime("%B")
        key = (location_name, month_name)
        if key in processed_keys:
            continue

        year = 2023
        user_prompt = USER_TEMPLATE.format(location_name=location_name, month=month_name, year=year, aod=aod)
        full_prompt = f"{SYSTEM_PROMPT}\n\n{user_prompt}"
        messages = [{"role": "user", "content": full_prompt}]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)

        batch_prompts.append(prompt)
        meta_info.append((location_name, month_name, year, aod))

        # when batch full → query
        if len(batch_prompts) == batch_size:
            preds = batch_query_llm(batch_prompts)
            for (location_name, month_name, year, aod), pred in zip(meta_info, preds):
                row = {
                    "location_name": location_name,
                    "year": year,
                    "month": month_name,
                    "aod": aod,
                    "model": model_id,
                    "pm2.5": pred,
                }
                rows.append(row)
                counter += 1
            batch_prompts, meta_info = [], []

            if counter % save_every == 0:
                pd.DataFrame(rows).to_csv(save_path, index=False)
                print(f"[{datetime.now()}] Saved at {counter} rows.")

    # leftover batch
    if batch_prompts:
        preds = batch_query_llm(batch_prompts)
        for (location_name, month_name, year, aod), pred in zip(meta_info, preds):
            row = {
                "location_name": location_name,
                "year": year,
                "month": month_name,
                "aod": aod,
                "model": model_id,
                "pm2.5": pred,
            }
            rows.append(row)
            counter += 1

# Final save
df_result = pd.DataFrame(rows)
df_result.to_csv(save_path, index=False)
