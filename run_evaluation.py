
import os
import subprocess

experiments = ["exp1", "exp2", "exp3", "exp4", "exp5"]
dataset_train = "train.csv"
dataset_test = "test.csv"
evaluate_script = "evaluate_model.py"
for exp in experiments:
    model_path = os.path.join("experiments", exp, "finetuned-gemma-classifier")
    if not os.path.exists(model_path):
        print(f"❌ Skipping {exp} — model path not found: {model_path}")
        continue
        
    # test_path = os.path.join("experiments", exp, "classification_report_test.csv")
    # if os.path.exists(test_path):
    #     print(f"❌ Skipping {exp} — test path  found: {test_path}")
    #     continue

    print(f"\n🚀 Running Evaluation for {exp}")

    # Evaluate on TRAIN data
    print(f"\n🚀 Running Evaluation for {exp} Train : ")
    subprocess.run([
        "python", evaluate_script,
        "--model_path", model_path,
        "--dataset_path", dataset_train,
        "--output_path", f"experiments/{exp}/classification_report_train.csv"
    ])

    # Evaluate on TEST data
    print(f"\n🚀 Running Evaluation for {exp} Test : ")
    subprocess.run([
        "python", evaluate_script,
        "--model_path", model_path,
        "--dataset_path", dataset_test,
        "--output_path", f"experiments/{exp}/classification_report_test.csv"
    ])
    print(f"\n🚀 Completed Evaluation for {exp} : ")
    
