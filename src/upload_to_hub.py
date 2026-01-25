#!/usr/bin/env python3
"""
Upload converted Arabic Mobile Actions dataset to Hugging Face Hub.
"""

import argparse
from pathlib import Path

from datasets import Dataset
from huggingface_hub import HfApi
import jsonlines


def load_jsonl(path: str) -> list:
    """Load JSONL file."""
    with jsonlines.open(path) as reader:
        return list(reader)


def upload_dataset(
    data_path: str,
    repo_id: str,
    readme_path: str = None,
    private: bool = False
):
    """
    Upload dataset to Hugging Face Hub.
    
    Args:
        data_path: Path to JSONL dataset file
        repo_id: HuggingFace repo ID (username/dataset-name)
        readme_path: Path to README.md for dataset card
        private: Whether to make the dataset private
    """
    print(f"Loading dataset from {data_path}...")
    data = load_jsonl(data_path)
    print(f"Loaded {len(data)} samples")
    
    # Create HuggingFace Dataset
    dataset = Dataset.from_list(data)
    print(f"Dataset created: {dataset}")
    
    # Push to Hub
    print(f"\nPushing to Hugging Face: {repo_id}")
    dataset.push_to_hub(
        repo_id,
        private=private,
        commit_message="Upload Arabic Mobile Actions dataset"
    )
    print("Dataset uploaded successfully!")
    
    # Upload README if provided
    if readme_path and Path(readme_path).exists():
        print(f"\nUploading README from {readme_path}...")
        api = HfApi()
        api.upload_file(
            path_or_fileobj=readme_path,
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type="dataset",
            commit_message="Add dataset card"
        )
        print("README uploaded!")
    
    print(f"\n✓ Dataset available at: https://huggingface.co/datasets/{repo_id}")


def main():
    parser = argparse.ArgumentParser(
        description="Upload Arabic Mobile Actions dataset to Hugging Face"
    )
    parser.add_argument(
        "--data", "-d",
        default="data/processed/arabic_mobile_actions.jsonl",
        help="Path to JSONL dataset"
    )
    parser.add_argument(
        "--repo", "-r",
        default="Sa74ll/arabic-mobile-actions",
        help="HuggingFace repo ID"
    )
    parser.add_argument(
        "--readme",
        default="data/processed/README.md",
        help="Path to README.md"
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Make dataset private"
    )
    
    args = parser.parse_args()
    
    upload_dataset(
        data_path=args.data,
        repo_id=args.repo,
        readme_path=args.readme,
        private=args.private
    )


if __name__ == "__main__":
    main()
