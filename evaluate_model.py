import pandas as pd
import re
import torch
from sklearn.metrics import classification_report
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel, PeftConfig
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str, required=True, help="Path to finetuned model directory")
parser.add_argument("--dataset_path", type=str, required=True, help="Path to CSV with prompts and labels")
parser.add_argument("--output_path", type=str, default="predicted_output.csv", help="Path to save predictions CSV")
args = parser.parse_args()

# === Load dataset ===
df = pd.read_csv(args.dataset_path)[["classification_prompt", "classification_label"]]
df["classification_prompt"] = df["classification_prompt"].astype(str).str.strip()
df["classification_label"] = df["classification_label"].astype(str).str.lower().str.strip()

# === Check for existing predictions to RESUME ===
if os.path.exists(args.output_path):
    prev = pd.read_csv(args.output_path)
    if "predicted" in prev.columns:
        df["predicted"] = prev["predicted"]
        print(f"🔄 Found {prev['predicted'].notna().sum()} completed predictions, will resume only unfinished.")
    else:
        df["predicted"] = None
else:
    df["predicted"] = None

# === Load PEFT config and base model ===
peft_config = PeftConfig.from_pretrained(args.model_path)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype="float16"
)
base_model = AutoModelForCausalLM.from_pretrained(
    peft_config.base_model_name_or_path,
    quantization_config=bnb_config,
    device_map="auto"
)
model = PeftModel.from_pretrained(base_model, args.model_path)
tokenizer = AutoTokenizer.from_pretrained(peft_config.base_model_name_or_path)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)
model.eval()

# === Inference only on unfinished samples ===
batch_size = 8
to_predict_idx = df[df["predicted"].isnull()].index.tolist()

print(f"\n⏳ Total: {len(df)} | Already predicted: {len(df) - len(to_predict_idx)} | To predict: {len(to_predict_idx)}")
for idx_start in range(0, len(to_predict_idx), batch_size):
    batch_idxs = to_predict_idx[idx_start:idx_start + batch_size]
    batch_prompts = df.loc[batch_idxs, "classification_prompt"].tolist()
    print(f"📊 Predicting batch {idx_start}–{idx_start+len(batch_idxs)-1} ...")

    inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True, max_length=128).to(device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=10)

    decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    for i, output in zip(batch_idxs, decoded):
        match = re.search(r"Category:\s*([\w\s\-]+)", output, re.IGNORECASE)
        label = match.group(1).strip().lower() if match else "general feedback"
        df.at[i, "predicted"] = label

    # --- Save after every batch (so you can interrupt safely!) ---
    df.to_csv(args.output_path, index=False)

print("\n✅ All predictions completed and saved.")

# === Map invalid predictions to "general feedback" ===
valid_labels = df["classification_label"].unique().tolist()
df["predicted"] = df["predicted"].apply(lambda x: x if x in valid_labels else "general feedback")

# === Classification report ===
report = classification_report(df["classification_label"], df["predicted"], labels=valid_labels, zero_division=0)
print("\n📊 Classification Report:")
print(report)

# === Save to .txt file ===
report_txt_path = args.output_path.replace(".csv", "_report.txt")
with open(report_txt_path, "w") as f:
    f.write("Classification Report\n")
    f.write("=====================\n")
    f.write(report)
print(f"\n📝 Classification report saved to {report_txt_path}")

# === Save results (final save for safety) ===
df.to_csv(args.output_path, index=False)
print(f"\n✅ Final results saved to {args.output_path}")
