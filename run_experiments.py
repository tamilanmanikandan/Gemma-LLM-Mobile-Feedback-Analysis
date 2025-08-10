
import os
import subprocess

# Define fast experiments (learning_rate >= 3e-5)
experiments = [
    {"lr": 5e-5, "epochs": 2, "r": 8, "alpha": 16, "dropout": 0.05},
    {"lr": 4e-5, "epochs": 2, "r": 8, "alpha": 32, "dropout": 0.1},
    {"lr": 3e-5, "epochs": 3, "r": 8, "alpha": 32, "dropout": 0.05},
    {"lr": 3e-5, "epochs": 2, "r": 16, "alpha": 16, "dropout": 0.1},
    {"lr": 3e-5, "epochs": 2, "r": 8, "alpha": 16, "dropout": 0.1}
]

for i, exp in enumerate(experiments, 1):
    output_dir = f"experiments/exp{i}"
    os.makedirs(output_dir, exist_ok=True)

    cmd = [
        "python", "finetune_gemma_lora.py",
        "--learning_rate", str(exp["lr"]),
        "--num_train_epochs", str(exp["epochs"]),
        "--lora_r", str(exp["r"]),
        "--lora_alpha", str(exp["alpha"]),
        "--lora_dropout", str(exp["dropout"]),
        "--output_dir", f"{output_dir}/model",
        "--model_dir", f"{output_dir}/finetuned-gemma-classifier",
        "--logging_dir", f"{output_dir}/logs",
        "--step_log_path", f"{output_dir}/step_logs.jsonl"
    ]

    print(f"🚀 Running Experiment {i}...")
    subprocess.run(cmd)
    print(f"✅ Completed Experiment {i}\n")
