
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--learning_rate", type=float, default=3e-5)
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.1)
    parser.add_argument("--per_device_train_batch_size", type=int, default=8)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2)
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--model_dir", type=str, default="finetuned-gemma-classifier")
    parser.add_argument("--logging_dir", type=str, default="logs")
    parser.add_argument("--step_log_path", type=str, default="step_logs.jsonl")
    return parser.parse_args()

args = parse_args()

from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import get_peft_model, LoraConfig, TaskType
from datasets import load_from_disk
from transformers import BitsAndBytesConfig
import os
# Load tokenized dataset
dataset = load_from_disk("tokenized_train")

# Load tokenizer and model
model_name = "google/gemma-2b"
tokenizer = AutoTokenizer.from_pretrained(model_name)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype="float16")


model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto"
    # load_in_4bit=True  # Use 4-bit for lower GPU memory
)

# Setup LoRA configuration
lora_config = LoraConfig(
    r=8,
    lora_alpha = args.lora_alpha,
    lora_dropout = args.lora_dropout,
    task_type=TaskType.CAUSAL_LM,
    bias="none",
    target_modules=["q_proj", "v_proj"]
)

model = get_peft_model(model, lora_config)

# Training configuration
training_args = TrainingArguments(
    output_dir = args.output_dir,
    num_train_epochs = args.num_train_epochs,
    per_device_train_batch_size = args.per_device_train_batch_size,
    gradient_accumulation_steps = args.gradient_accumulation_steps,
    learning_rate = args.learning_rate,
    eval_strategy="no",
    # save_strategy="epoch",
    save_strategy="steps",
    save_steps=1000,
    logging_steps=20,
    logging_dir=args.logging_dir,
    fp16=True  # Use float16 if supported
)


from transformers import TrainerCallback
import json

class StepLoggerCallback(TrainerCallback):
    def __init__(self, log_file="" + args.step_log_path + ""):
        self.log_file = log_file
        with open(self.log_file, "w") as f:
            pass  # clear existing content

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            with open(self.log_file, "a") as f:
                f.write(json.dumps(logs) + "\n")

                
# Trainer
trainer = Trainer(
    callbacks=[StepLoggerCallback("" + args.step_log_path + "")],
    model=model,
    args=training_args,
    train_dataset=dataset,
    tokenizer=tokenizer
)
# Attach callback to the trainer
trainer.add_callback(StepLoggerCallback("" + args.step_log_path + ""))



# checkpoint_dir = "./gemma-lora-output"
checkpoint_dir = args.output_dir
checkpoint_path = None

# Find latest checkpoint
if os.path.isdir(checkpoint_dir):
    checkpoints = [d for d in os.listdir(checkpoint_dir) if d.startswith("checkpoint-")]
    if checkpoints:
        checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[-1]))
        checkpoint_path = os.path.join(checkpoint_dir, checkpoints[-1])
        print(f"📍 Found checkpoint: {checkpoint_path}")
    else:
        print("⚠️ No checkpoints found in output directory.")
else:
    print("⚠️ Checkpoint directory does not exist.")

# === Fine-tune with resume option ===
if checkpoint_path:
    trainer.train(resume_from_checkpoint=checkpoint_path)
else:
    trainer.train()
# Fine-tuning
# trainer.train()

# Save model
model.save_pretrained(args.model_dir)
tokenizer.save_pretrained(args.model_dir)

print("✅ Fine-tuning complete. Model saved to 'finetuned-gemma-classifier'")


# # === Generate training set classification report ===
# from sklearn.metrics import classification_report

# # Reload tokenized dataset if needed
# from datasets import load_from_disk
# tokenized_dataset = load_from_disk("tokenized_train")

# # Optional: reduce sample size to avoid OOM
# sample_dataset = tokenized_dataset.select(range(min(10000, len(tokenized_dataset))))

# # Predict
# outputs = trainer.predict(sample_dataset)
# preds = tokenizer.batch_decode(outputs.predictions, skip_special_tokens=True)

# # Clean up prediction format
# def clean_prediction(pred_str):
#     pred_str = pred_str.lower().strip()
#     for label in ["feature request", "bug report", "general feedback"]:
#         if label in pred_str:
#             return label
#     return "unknown"

# # Convert to labels
# label_map = {0: "feature request", 1: "bug report", 2: "general feedback"}
# y_pred = [clean_prediction(p) for p in preds]
# y_true = [label_map[int(l)] for l in outputs.label_ids]

# # Report
# report = classification_report(y_true, y_pred, digits=2)
# print("\n📊 Training Classification Report:\n")
# print(report)

# # Save to file
# with open("training_classification_report.txt", "w") as f:
#     f.write(report)
