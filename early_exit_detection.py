import torch
import numpy as np
import random
from pathlib import Path
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_from_disk
from lib.model_utils import load_model_config

def to_dataset(dataset_or_dict):
    """Extract a single Dataset from a DatasetDict or dict wrapper if needed."""
    if hasattr(dataset_or_dict, "column_names") and isinstance(dataset_or_dict.column_names, dict):
        if "train" in dataset_or_dict:
            return dataset_or_dict["train"]
        return dataset_or_dict[list(dataset_or_dict.keys())[0]]
    if hasattr(dataset_or_dict, "keys") and not hasattr(dataset_or_dict, "column_names"):
        if "train" in dataset_or_dict:
            return dataset_or_dict["train"]
        return dataset_or_dict[list(dataset_or_dict.keys())[0]]
    return dataset_or_dict


def main():
    # ---------------------------------------------------------
    # 1. Load configured model and reconstruct its harm vector
    # ---------------------------------------------------------
    config_path = Path(__file__).resolve().parent / "config" / "config.yml"
    model_id, _, auc_layer = load_model_config(config_path)
    num_layers = auc_layer

    print("Loading cached layer data to calibrate the radar...")
    data = np.load(f"./data/{model_id}_all_layers.npz")

    X_benign = data["benign_states"][:, auc_layer, :]
    X_malicious = data["malicious_states"][:, auc_layer, :]

    calib_size = 50
    mu_benign = np.mean(X_benign[:calib_size], axis=0)
    mu_malicious = np.mean(X_malicious[:calib_size], axis=0)

    harm_vector = mu_malicious - mu_benign
    harm_vector = harm_vector / np.linalg.norm(harm_vector)
    threshold = np.percentile(np.dot(X_benign, harm_vector), 99)
    print(f"Layer {auc_layer} Tripwire Threshold established at: {threshold:.4f}\n")

    # ---------------------------------------------------------
    # 2. Early-Exit Model Optimization
    # ---------------------------------------------------------
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    print(f"Loading {model_id} and retaining {num_layers} layers for early-exit...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
    ).to(device)
    model.model.layers = model.model.layers[:num_layers]
    print(f"Sliced model to {num_layers} layers (hidden state {auc_layer}).\n")

    def score_prompt_early_exit(prompt):
        formatted = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = tokenizer(
            formatted,
            return_tensors="pt",
            truncation=True,
            max_length=1024,
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)

        raw_vector = outputs.hidden_states[-1][0, -1, :].float().cpu().numpy()
        normalized_vector = raw_vector / np.linalg.norm(raw_vector)
        return np.dot(normalized_vector, harm_vector)

    # ---------------------------------------------------------
    # 3. Generate Stress Tests (Dynamic Column Resolution)
    # ---------------------------------------------------------
    print("Loading stress test datasets from local disk...")

    # --- Benign Prompts (UltraChat) ---
    ultrachat_dataset = to_dataset(load_from_disk("./data/ultrachat"))
    benign_sample_size = min(100, len(ultrachat_dataset))
    if "prompt" in ultrachat_dataset.column_names:
        benign_prompts = list(ultrachat_dataset["prompt"][:benign_sample_size])
    elif "messages" in ultrachat_dataset.column_names:
        benign_prompts = [
            msg[0]["content"] if isinstance(msg, list) and isinstance(msg[0], dict) else msg
            for msg in ultrachat_dataset["messages"][:benign_sample_size]
        ]
    else:
        benign_col = ultrachat_dataset.column_names[0]
        benign_prompts = list(ultrachat_dataset[benign_col][:benign_sample_size])

    # --- Base Malicious Attacks (HarmBench) ---
    harmbench_dataset = to_dataset(load_from_disk("./data/harmbench"))
    hb_columns = harmbench_dataset.column_names
    hb_col = next((col for col in ["Behavior", "behavior", "prompt"] if col in hb_columns), hb_columns[0])
    base_attacks = list(harmbench_dataset[hb_col][:100])
    malicious_prompts = base_attacks

    # --- GCG Gibberish Suffixes ---
    gcg_dataset = to_dataset(load_from_disk("./data/adv_suffixes"))
    print(f"GCG Columns detected: {gcg_dataset.column_names}")
    gcg_col = next((col for col in ["adv_suffix", "adversarial_suffix", "suffix", "text", "prompt"] if col in gcg_dataset.column_names), gcg_dataset.column_names[-1])

    # Cap sample size to prevent ValueError
    gcg_sample_size = min(100, len(gcg_dataset), len(base_attacks))
    random.seed(42)
    sampled_suffixes = random.sample(gcg_dataset[gcg_col], gcg_sample_size)
    gcg_prompts = [f"{attack} {suffix}" for attack, suffix in zip(base_attacks[:gcg_sample_size], sampled_suffixes)]

    # --- Roleplay / Developer Mode Wrappers ---
    dan_dataset = to_dataset(load_from_disk("./data/jailbreak_prompts"))
    print(f"DAN Columns detected: {dan_dataset.column_names}")
    dan_col = next((col for col in ["Prompt", "prompt", "text"] if col in dan_dataset.column_names), dan_dataset.column_names[0])

    # Cap sample size to prevent ValueError
    dan_sample_size = min(100, len(dan_dataset), len(base_attacks))
    sampled_jailbreaks = random.sample(dan_dataset[dan_col], dan_sample_size)
    roleplay_prompts = [f"{jailbreak}\n\nTask: {attack}" for attack, jailbreak in zip(base_attacks[:dan_sample_size], sampled_jailbreaks)]

    print(f"\nLoaded {len(benign_prompts)} benign prompts, {len(malicious_prompts)} malicious base attacks, {len(gcg_prompts)} GCG attacks, and {len(roleplay_prompts)} Roleplay attacks.")

    # ---------------------------------------------------------
    # 4. Evaluate Stress Tests
    # ---------------------------------------------------------
    print("\nTesting Benign Prompts...")
    benign_passed = sum(
        score_prompt_early_exit(prompt) <= threshold
        for prompt in tqdm(benign_prompts)
    )

    print("\nTesting Malicious Base Attacks...")
    malicious_caught = sum(
        score_prompt_early_exit(prompt) > threshold
        for prompt in tqdm(malicious_prompts)
    )

    print("\nTesting GCG Suffix Attacks...")
    gcg_caught = sum(
        score_prompt_early_exit(prompt) > threshold
        for prompt in tqdm(gcg_prompts)
    )

    print("\nTesting Roleplay/DAN Attacks...")
    roleplay_caught = sum(
        score_prompt_early_exit(prompt) > threshold
        for prompt in tqdm(roleplay_prompts)
    )

    print("\n" + "=" * 40)
    print(f"STRESS TEST RESULTS (LAYER {auc_layer} EARLY-EXIT)")
    print("=" * 40)
    print(f"Benign Prompts Passed:            {benign_passed} / {len(benign_prompts)}")
    print(f"Base Attacks Blocked:             {malicious_caught} / {len(malicious_prompts)}")
    print(f"GCG Suffixes Blocked:             {gcg_caught} / {len(gcg_prompts)}")
    print(f"Roleplay Wrappers Blocked:        {roleplay_caught} / {len(roleplay_prompts)}")
    print("=" * 40)

    return {
        "benign_passed": benign_passed,
        "benign_total": len(benign_prompts),
        "malicious_caught": malicious_caught,
        "malicious_total": len(malicious_prompts),
        "gcg_caught": gcg_caught,
        "gcg_total": len(gcg_prompts),
        "roleplay_caught": roleplay_caught,
        "roleplay_total": len(roleplay_prompts),
    }


if __name__ == "__main__":
    main()