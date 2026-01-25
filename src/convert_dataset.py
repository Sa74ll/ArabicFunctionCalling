#!/usr/bin/env python3
"""
Convert Arabic Function Calling dataset to Google Mobile Actions format.

Source: HeshamHaroon/Arabic_Function_Calling
Target: google/mobile-actions format for FunctionGemma fine-tuning
"""

import json
import argparse
import uuid
from pathlib import Path
from typing import Optional

from datasets import load_dataset
from tqdm import tqdm
import jsonlines


def load_function_schemas(schema_path: str) -> dict:
    """Load function schemas from JSON file."""
    with open(schema_path, 'r', encoding='utf-8') as f:
        schemas = json.load(f)
    
    # Flatten schemas into a function_name -> schema mapping
    flat_schemas = {}
    for domain, functions in schemas.items():
        for func_name, func_schema in functions.items():
            flat_schemas[func_name] = func_schema
    
    return flat_schemas


def convert_sample(
    sample: dict,
    function_schemas: dict,
    split: str = "train"
) -> Optional[dict]:
    """
    Convert a single Arabic Function Calling sample to Mobile Actions format.
    
    Args:
        sample: Original dataset sample
        function_schemas: Dictionary of function schemas
        split: Dataset split (train/eval)
    
    Returns:
        Converted sample in Mobile Actions format, or None if conversion fails
    """
    function_name = sample.get("function_name")
    requires_function = sample.get("requires_function", True)
    query_ar = sample.get("query_ar", "")
    arguments = sample.get("arguments", {})
    
    # Build the messages array with developer role for FunctionGemma
    messages = [
        {
            "role": "developer",
            "content": "You are a helpful assistant that can use tools to help the user. When the user asks for something that requires a tool, call the appropriate function."
        },
        {
            "role": "user",
            "content": query_ar
        }
    ]
    
    # Build tools array - include the function schema if available
    tools = []
    if function_name and function_name in function_schemas:
        tools.append(function_schemas[function_name])
    elif function_name:
        # Create a basic schema if not in registry
        tools.append({
            "function": {
                "name": function_name,
                "description": f"Function: {function_name}",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": []
                }
            }
        })
    
    # Build assistant response
    if requires_function and function_name:
        # Positive sample - include tool call
        tool_call_id = f"call_{uuid.uuid4().hex[:8]}"
        
        # Handle arguments - they may already be a string or a dict
        if isinstance(arguments, str):
            arguments_str = arguments
        else:
            arguments_str = json.dumps(arguments, ensure_ascii=False)
        
        assistant_message = {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": tool_call_id,
                    "type": "function",
                    "function": {
                        "name": function_name,
                        "arguments": arguments_str
                    }
                }
            ]
        }
    else:
        # Negative sample - no tool call, just empty/decline response
        assistant_message = {
            "role": "assistant",
            "content": ""  # Model should not call any function
        }
    
    messages.append(assistant_message)
    
    # Build the final Mobile Actions format sample
    converted = {
        "metadata": split,
        "tools": tools,
        "messages": messages
    }
    
    # Add dialect and domain as extra metadata (useful for evaluation)
    if "dialect" in sample:
        converted["dialect"] = sample["dialect"]
    if "domain" in sample:
        converted["domain"] = sample["domain"]
    if "id" in sample:
        converted["original_id"] = sample["id"]
    
    return converted


def convert_dataset(
    output_path: str,
    schema_path: str,
    train_ratio: float = 0.8,
    limit: Optional[int] = None,
    seed: int = 42
):
    """
    Convert the entire Arabic Function Calling dataset.
    
    Args:
        output_path: Path to save the converted JSONL file
        schema_path: Path to function schemas JSON file
        train_ratio: Ratio of samples for training (rest goes to eval)
        limit: Optional limit on number of samples to convert
        seed: Random seed for train/eval split
    """
    print("Loading function schemas...")
    function_schemas = load_function_schemas(schema_path)
    print(f"Loaded {len(function_schemas)} function schemas")
    
    print("\nLoading Arabic Function Calling dataset from HuggingFace...")
    dataset = load_dataset("HeshamHaroon/Arabic_Function_Calling", split="train")
    
    if limit:
        dataset = dataset.select(range(min(limit, len(dataset))))
    
    print(f"Dataset size: {len(dataset)} samples")
    
    # Shuffle and split
    dataset = dataset.shuffle(seed=seed)
    train_size = int(len(dataset) * train_ratio)
    
    # Track statistics
    stats = {
        "total": 0,
        "converted": 0,
        "train": 0,
        "eval": 0,
        "positive": 0,
        "negative": 0,
        "by_dialect": {},
        "by_domain": {},
        "by_function": {}
    }
    
    print(f"\nConverting to Mobile Actions format...")
    print(f"Train/Eval split: {train_ratio:.0%} / {1-train_ratio:.0%}")
    
    # Ensure output directory exists
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    with jsonlines.open(output_path, mode='w') as writer:
        for idx, sample in enumerate(tqdm(dataset, desc="Converting")):
            stats["total"] += 1
            
            # Determine split
            split = "train" if idx < train_size else "eval"
            
            # Convert sample
            converted = convert_sample(sample, function_schemas, split)
            
            if converted:
                writer.write(converted)
                stats["converted"] += 1
                stats[split] += 1
                
                # Track positive/negative
                if sample.get("requires_function", True):
                    stats["positive"] += 1
                else:
                    stats["negative"] += 1
                
                # Track by dialect
                dialect = sample.get("dialect", "unknown")
                stats["by_dialect"][dialect] = stats["by_dialect"].get(dialect, 0) + 1
                
                # Track by domain
                domain = sample.get("domain", "unknown")
                stats["by_domain"][domain] = stats["by_domain"].get(domain, 0) + 1
                
                # Track by function
                func = sample.get("function_name", "none")
                stats["by_function"][func] = stats["by_function"].get(func, 0) + 1
    
    # Print statistics
    print("\n" + "="*60)
    print("CONVERSION COMPLETE")
    print("="*60)
    print(f"Output: {output_path}")
    print(f"\nTotal samples: {stats['total']}")
    print(f"Successfully converted: {stats['converted']}")
    print(f"  - Train: {stats['train']}")
    print(f"  - Eval: {stats['eval']}")
    print(f"\nSample types:")
    print(f"  - Positive (with function call): {stats['positive']}")
    print(f"  - Negative (no function call): {stats['negative']}")
    
    print(f"\nBy dialect:")
    for dialect, count in sorted(stats["by_dialect"].items(), key=lambda x: -x[1]):
        print(f"  - {dialect}: {count}")
    
    print(f"\nBy domain:")
    for domain, count in sorted(stats["by_domain"].items(), key=lambda x: -x[1]):
        print(f"  - {domain}: {count}")
    
    print(f"\nTop 10 functions:")
    for func, count in sorted(stats["by_function"].items(), key=lambda x: -x[1])[:10]:
        print(f"  - {func}: {count}")
    
    # Save statistics
    stats_path = output_path.replace('.jsonl', '_stats.json')
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(f"\nStatistics saved to: {stats_path}")
    
    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Convert Arabic Function Calling dataset to Mobile Actions format"
    )
    parser.add_argument(
        "--output", "-o",
        default="data/processed/arabic_mobile_actions.jsonl",
        help="Output JSONL file path"
    )
    parser.add_argument(
        "--schemas", "-s",
        default="schemas/function_schemas.json",
        help="Path to function schemas JSON"
    )
    parser.add_argument(
        "--train-ratio", "-t",
        type=float,
        default=0.8,
        help="Ratio of samples for training (default: 0.8)"
    )
    parser.add_argument(
        "--limit", "-l",
        type=int,
        default=None,
        help="Limit number of samples to convert (for testing)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    
    args = parser.parse_args()
    
    convert_dataset(
        output_path=args.output,
        schema_path=args.schemas,
        train_ratio=args.train_ratio,
        limit=args.limit,
        seed=args.seed
    )


if __name__ == "__main__":
    main()
