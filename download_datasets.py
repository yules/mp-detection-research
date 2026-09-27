import os
import pandas as pd
from datasets import load_dataset

ADVBENCH_URL = (
    "https://raw.githubusercontent.com/llm-attacks/llm-attacks/main/"
    "data/advbench/harmful_behaviors.csv"
)


def _download_and_save(dataset_id, output_path, *load_args, **load_kwargs):
    print(f"Downloading {dataset_id}...")
    try:
        dataset = load_dataset(dataset_id, *load_args, **load_kwargs)
        dataset.save_to_disk(output_path)
        print(f"Successfully saved {dataset_id} to: {output_path}")
    except Exception as error:
        print(f"Failed to download {dataset_id}: {error}")


def _download_advbench(output_path):
    print("Downloading AdvBench...")
    try:
        advbench = pd.read_csv(ADVBENCH_URL)
        advbench.to_csv(output_path, index=False)
        print(f"Successfully saved AdvBench to: {output_path}")
    except Exception as error:
        print(f"Failed to download AdvBench: {error}")


def download_and_save_datasets(base_dir="./data"):
    """
    Downloads baseline and stress-test datasets and saves them locally.
    """
    os.makedirs(base_dir, exist_ok=True)

    _download_and_save(
        "walledai/HarmBench",
        os.path.join(base_dir, "harmbench"),
        "standard",
        trust_remote_code=True,
    )
    _download_and_save(
        "FloofCat/AdvSuffixes",
        os.path.join(base_dir, "adv_suffixes"),
    )
    _download_and_save(
        "rubend18/ChatGPT-Jailbreak-Prompts",
        os.path.join(base_dir, "jailbreak_prompts"),
    )
    _download_advbench(os.path.join(base_dir, "advbench.csv"))

    print("\nAll downloads complete! Your datasets are ready.")

if __name__ == "__main__":
    # Ensure you have 'datasets' installed: pip install datasets
    download_and_save_datasets()