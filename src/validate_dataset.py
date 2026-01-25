#!/usr/bin/env python3
"""
Validate converted Arabic Mobile Actions dataset.

Checks:
1. JSON structure matches Mobile Actions format
2. Tokenizer can process all samples
3. Statistics and quality checks
"""

import json
import argparse
from pathlib import Path
from collections import defaultdict

import jsonlines
from tqdm import tqdm
from rich.console import Console
from rich.table import Table

console = Console()


def validate_sample_structure(sample: dict, idx: int) -> list:
    """Validate a single sample's structure."""
    errors = []
    
    # Required fields
    if "metadata" not in sample:
        errors.append(f"Sample {idx}: Missing 'metadata' field")
    elif sample["metadata"] not in ["train", "eval"]:
        errors.append(f"Sample {idx}: Invalid metadata value: {sample['metadata']}")
    
    if "tools" not in sample:
        errors.append(f"Sample {idx}: Missing 'tools' field")
    elif not isinstance(sample["tools"], list):
        errors.append(f"Sample {idx}: 'tools' must be a list")
    else:
        for i, tool in enumerate(sample["tools"]):
            if "function" not in tool:
                errors.append(f"Sample {idx}, tool {i}: Missing 'function' field")
            else:
                func = tool["function"]
                if "name" not in func:
                    errors.append(f"Sample {idx}, tool {i}: Missing function 'name'")
                if "parameters" not in func:
                    errors.append(f"Sample {idx}, tool {i}: Missing function 'parameters'")
    
    if "messages" not in sample:
        errors.append(f"Sample {idx}: Missing 'messages' field")
    elif not isinstance(sample["messages"], list):
        errors.append(f"Sample {idx}: 'messages' must be a list")
    elif len(sample["messages"]) < 2:
        errors.append(f"Sample {idx}: 'messages' must have at least 2 messages")
    else:
        # Check user message
        if sample["messages"][0].get("role") != "user":
            errors.append(f"Sample {idx}: First message must be from 'user'")
        if "content" not in sample["messages"][0]:
            errors.append(f"Sample {idx}: User message missing 'content'")
        
        # Check assistant message
        if sample["messages"][1].get("role") != "assistant":
            errors.append(f"Sample {idx}: Second message must be from 'assistant'")
        
        assistant_msg = sample["messages"][1]
        has_tool_calls = "tool_calls" in assistant_msg
        has_content = "content" in assistant_msg
        
        if not has_tool_calls and not has_content:
            errors.append(f"Sample {idx}: Assistant message must have 'tool_calls' or 'content'")
        
        if has_tool_calls:
            for j, call in enumerate(assistant_msg["tool_calls"]):
                if "id" not in call:
                    errors.append(f"Sample {idx}, call {j}: Missing 'id'")
                if call.get("type") != "function":
                    errors.append(f"Sample {idx}, call {j}: type must be 'function'")
                if "function" not in call:
                    errors.append(f"Sample {idx}, call {j}: Missing 'function'")
                else:
                    if "name" not in call["function"]:
                        errors.append(f"Sample {idx}, call {j}: Missing function 'name'")
                    if "arguments" not in call["function"]:
                        errors.append(f"Sample {idx}, call {j}: Missing function 'arguments'")
                    elif not isinstance(call["function"]["arguments"], str):
                        errors.append(f"Sample {idx}, call {j}: 'arguments' must be a string")
    
    return errors


def validate_with_tokenizer(sample: dict, tokenizer) -> list:
    """Validate sample can be processed by tokenizer."""
    errors = []
    
    try:
        # Try to apply chat template
        text = tokenizer.apply_chat_template(
            sample["messages"],
            tools=sample["tools"],
            tokenize=False
        )
        
        # Try to tokenize
        tokens = tokenizer.encode(text, add_special_tokens=True)
        
        if len(tokens) > tokenizer.model_max_length:
            errors.append(f"Token length ({len(tokens)}) exceeds max ({tokenizer.model_max_length})")
            
    except Exception as e:
        errors.append(f"Tokenizer error: {str(e)}")
    
    return errors


def validate_dataset(
    input_path: str,
    use_tokenizer: bool = False,
    model_name: str = "google/gemma-2b-it",
    sample_limit: int = None
):
    """
    Validate the converted dataset.
    
    Args:
        input_path: Path to the converted JSONL file
        use_tokenizer: Whether to validate with tokenizer
        model_name: Model name for tokenizer
        sample_limit: Limit validation to N samples
    """
    console.print(f"\n[bold blue]Validating dataset: {input_path}[/bold blue]\n")
    
    # Load tokenizer if requested
    tokenizer = None
    if use_tokenizer:
        console.print("Loading tokenizer...")
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        console.print(f"Loaded tokenizer: {model_name}")
    
    # Statistics
    stats = {
        "total": 0,
        "valid": 0,
        "errors": 0,
        "train": 0,
        "eval": 0,
        "with_tool_calls": 0,
        "without_tool_calls": 0,
        "by_dialect": defaultdict(int),
        "by_domain": defaultdict(int),
        "by_function": defaultdict(int),
        "token_lengths": []
    }
    
    all_errors = []
    
    # Read and validate
    with jsonlines.open(input_path) as reader:
        samples = list(reader)
    
    if sample_limit:
        samples = samples[:sample_limit]
    
    console.print(f"Validating {len(samples)} samples...\n")
    
    for idx, sample in enumerate(tqdm(samples, desc="Validating")):
        stats["total"] += 1
        
        # Structure validation
        errors = validate_sample_structure(sample, idx)
        
        # Tokenizer validation
        if tokenizer and not errors:
            tok_errors = validate_with_tokenizer(sample, tokenizer)
            errors.extend(tok_errors)
        
        if errors:
            stats["errors"] += 1
            all_errors.extend(errors[:3])  # Limit errors per sample
        else:
            stats["valid"] += 1
        
        # Collect statistics
        stats[sample.get("metadata", "unknown")] += 1
        
        if sample.get("dialect"):
            stats["by_dialect"][sample["dialect"]] += 1
        if sample.get("domain"):
            stats["by_domain"][sample["domain"]] += 1
        
        # Check for tool calls
        if len(sample.get("messages", [])) > 1:
            assistant_msg = sample["messages"][1]
            if "tool_calls" in assistant_msg and assistant_msg["tool_calls"]:
                stats["with_tool_calls"] += 1
                for call in assistant_msg["tool_calls"]:
                    func_name = call.get("function", {}).get("name", "unknown")
                    stats["by_function"][func_name] += 1
            else:
                stats["without_tool_calls"] += 1
    
    # Print results
    console.print("\n" + "="*60)
    console.print("[bold green]VALIDATION RESULTS[/bold green]")
    console.print("="*60)
    
    # Summary table
    summary_table = Table(title="Summary")
    summary_table.add_column("Metric", style="cyan")
    summary_table.add_column("Value", style="green")
    
    summary_table.add_row("Total samples", str(stats["total"]))
    summary_table.add_row("Valid samples", str(stats["valid"]))
    summary_table.add_row("Samples with errors", str(stats["errors"]))
    summary_table.add_row("Train samples", str(stats["train"]))
    summary_table.add_row("Eval samples", str(stats["eval"]))
    summary_table.add_row("With tool calls", str(stats["with_tool_calls"]))
    summary_table.add_row("Without tool calls (negative)", str(stats["without_tool_calls"]))
    
    console.print(summary_table)
    
    # Dialect distribution
    if stats["by_dialect"]:
        dialect_table = Table(title="Dialect Distribution")
        dialect_table.add_column("Dialect", style="cyan")
        dialect_table.add_column("Count", style="green")
        dialect_table.add_column("Percentage", style="yellow")
        
        for dialect, count in sorted(stats["by_dialect"].items(), key=lambda x: -x[1]):
            pct = count / stats["total"] * 100
            dialect_table.add_row(dialect, str(count), f"{pct:.1f}%")
        
        console.print(dialect_table)
    
    # Domain distribution
    if stats["by_domain"]:
        domain_table = Table(title="Domain Distribution")
        domain_table.add_column("Domain", style="cyan")
        domain_table.add_column("Count", style="green")
        
        for domain, count in sorted(stats["by_domain"].items(), key=lambda x: -x[1]):
            domain_table.add_row(domain, str(count))
        
        console.print(domain_table)
    
    # Top functions
    if stats["by_function"]:
        func_table = Table(title="Top 10 Functions")
        func_table.add_column("Function", style="cyan")
        func_table.add_column("Count", style="green")
        
        for func, count in sorted(stats["by_function"].items(), key=lambda x: -x[1])[:10]:
            func_table.add_row(func, str(count))
        
        console.print(func_table)
    
    # Print errors
    if all_errors:
        console.print("\n[bold red]Sample Errors (first 10):[/bold red]")
        for error in all_errors[:10]:
            console.print(f"  • {error}")
    
    # Final verdict
    console.print("\n" + "="*60)
    if stats["errors"] == 0:
        console.print("[bold green]✓ VALIDATION PASSED[/bold green]")
    else:
        console.print(f"[bold yellow]⚠ VALIDATION COMPLETE WITH {stats['errors']} ERRORS[/bold yellow]")
    console.print("="*60)
    
    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Validate converted Arabic Mobile Actions dataset"
    )
    parser.add_argument(
        "--input", "-i",
        default="data/processed/arabic_mobile_actions.jsonl",
        help="Input JSONL file path"
    )
    parser.add_argument(
        "--tokenizer", "-t",
        action="store_true",
        help="Validate with tokenizer"
    )
    parser.add_argument(
        "--model", "-m",
        default="google/gemma-2b-it",
        help="Model name for tokenizer"
    )
    parser.add_argument(
        "--limit", "-l",
        type=int,
        default=None,
        help="Limit validation to N samples"
    )
    
    args = parser.parse_args()
    
    validate_dataset(
        input_path=args.input,
        use_tokenizer=args.tokenizer,
        model_name=args.model,
        sample_limit=args.limit
    )


if __name__ == "__main__":
    main()
