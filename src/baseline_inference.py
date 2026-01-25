#!/usr/bin/env python3
"""
Phase 2: Baseline Inference for FunctionGemma on Arabic Function Calling.

Tests pretrained FunctionGemma zero-shot on Arabic prompts to establish baseline metrics.
"""

import json
import argparse
import random
from pathlib import Path
from datetime import datetime
from collections import defaultdict

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
from rich.console import Console
from rich.table import Table

console = Console()

# Set device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_model_and_tokenizer(model_name: str = "google/gemma-2b-it"):
    """Load FunctionGemma model and tokenizer."""
    console.print(f"\n[bold blue]Loading model: {model_name}[/bold blue]")
    console.print(f"Device: {DEVICE}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Load model with appropriate settings for GPU
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.bfloat16 if DEVICE == "cuda" else torch.float32,
        low_cpu_mem_usage=True,
    )
    
    console.print(f"[green]✓ Model loaded successfully[/green]")
    return model, tokenizer


def format_tools_prompt(tools: list) -> str:
    """Format tools for the prompt."""
    tools_str = json.dumps(tools, ensure_ascii=False, indent=2)
    return f"Available tools:\n{tools_str}"


def clean_tools_for_functiongemma(tools: list) -> list:
    """
    Clean and format tools for FunctionGemma.
    - Adds 'type': 'function' wrapper if missing
    - Removes null properties from parameters
    """
    cleaned_tools = []
    for tool in tools:
        # Handle both formats: {"function": {...}} and {"type": "function", "function": {...}}
        if "function" in tool:
            func_def = tool["function"]
        else:
            func_def = tool

        # Clean up parameters - remove null values
        if "parameters" in func_def and "properties" in func_def["parameters"]:
            cleaned_properties = {
                k: v for k, v in func_def["parameters"]["properties"].items()
                if v is not None
            }
            func_def = dict(func_def)  # Make a copy
            func_def["parameters"] = dict(func_def["parameters"])
            func_def["parameters"]["properties"] = cleaned_properties

        # Ensure proper format with type wrapper
        cleaned_tool = {
            "type": "function",
            "function": func_def
        }
        cleaned_tools.append(cleaned_tool)

    return cleaned_tools


def generate_response(
    model,
    tokenizer,
    user_query: str,
    tools: list,
    max_new_tokens: int = 256,
    temperature: float = 0.1,
) -> str:
    """Generate model response for a given query and tools."""

    # Clean tools for FunctionGemma format
    cleaned_tools = clean_tools_for_functiongemma(tools)

    # Build messages with developer role for FunctionGemma
    # The developer message is CRITICAL to activate function calling mode
    messages = [
        {
            "role": "developer",
            "content": "You are a helpful assistant that can call functions to help users. When a user request requires using a tool, output the appropriate function call."
        },
        {
            "role": "user",
            "content": user_query
        }
    ]

    # Apply chat template with tools passed correctly
    # FunctionGemma expects tools in the native format
    try:
        prompt = tokenizer.apply_chat_template(
            messages,
            tools=cleaned_tools,  # Pass cleaned tools to the template
            tokenize=False,
            add_generation_prompt=True
        )
    except TypeError:
        # Fallback for models that don't support tools parameter
        tools_prompt = f"Available tools:\n{json.dumps(cleaned_tools, ensure_ascii=False, indent=2)}"
        messages[1]["content"] = f"{tools_prompt}\n\nUser request: {user_query}"
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
    
    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=temperature > 0,
            pad_token_id=tokenizer.eos_token_id,
        )
    
    # Decode response (only the generated part)
    response = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True
    )
    
    return response.strip()


def extract_function_call(response: str) -> dict:
    """Try to extract function call from model response."""
    result = {
        "raw_response": response,
        "function_name": None,
        "arguments": None,
        "has_function_call": False,
        "parse_error": None
    }

    # Try to find function call in response
    try:
        import re

        # Pattern 1: FunctionGemma format
        # <start_function_call>call:function_name{key:<escape>value<escape>}<end_function_call>
        # Also handles: call function_name{...} (with space instead of colon)
        functiongemma_match = re.search(
            r'<start_function_call>call[:\s]+(\w+)\{(.*?)\}<end_function_call>',
            response,
            re.DOTALL
        )
        if functiongemma_match:
            result["function_name"] = functiongemma_match.group(1)
            result["has_function_call"] = True
            # Parse FunctionGemma arguments
            args_str = functiongemma_match.group(2)
            args = {}
            # Match key:<escape>value<escape> patterns (string values)
            arg_matches = re.findall(r'(\w+):<escape>(.*?)<escape>', args_str)
            for key, value in arg_matches:
                args[key] = value
            # Also match key:number patterns (numeric values like days:7)
            num_matches = re.findall(r'(\w+):(\d+(?:\.\d+)?)', args_str)
            for key, value in num_matches:
                if key not in args:  # Don't overwrite escaped string values
                    args[key] = value
            result["arguments"] = args
            return result

        # Pattern 1b: FunctionGemma without end tag (partial generation)
        functiongemma_partial = re.search(
            r'<start_function_call>call[:\s]+(\w+)\{(.*?)\}',
            response,
            re.DOTALL
        )
        if functiongemma_partial:
            result["function_name"] = functiongemma_partial.group(1)
            result["has_function_call"] = True
            args_str = functiongemma_partial.group(2)
            args = {}
            arg_matches = re.findall(r'(\w+):<escape>(.*?)<escape>', args_str)
            for key, value in arg_matches:
                args[key] = value
            num_matches = re.findall(r'(\w+):(\d+(?:\.\d+)?)', args_str)
            for key, value in num_matches:
                if key not in args:
                    args[key] = value
            result["arguments"] = args
            return result

        # Pattern 2: Direct JSON object with "name" field
        json_match = re.search(r'\{[^{}]*"name"[^{}]*\}', response)
        if json_match:
            parsed = json.loads(json_match.group())
            if "name" in parsed:
                result["function_name"] = parsed.get("name")
                result["arguments"] = parsed.get("arguments", {})
                result["has_function_call"] = True
                return result

        # Pattern 3: Function name followed by arguments in parentheses
        func_match = re.search(r'(\w+)\s*\((.*?)\)', response)
        if func_match:
            result["function_name"] = func_match.group(1)
            result["has_function_call"] = True
            try:
                result["arguments"] = json.loads(func_match.group(2))
            except:
                result["arguments"] = func_match.group(2)
            return result

        # Pattern 4: Look for tool_calls structure
        if "tool_calls" in response.lower() or "function" in response.lower():
            result["has_function_call"] = True
            result["parse_error"] = "Could not parse function call structure"

    except Exception as e:
        result["parse_error"] = str(e)

    return result


def evaluate_prediction(prediction: dict, expected: dict) -> dict:
    """Evaluate a single prediction against expected output."""
    result = {
        "function_correct": False,
        "arguments_correct": False,
        "no_call_correct": False,  # For negative samples
    }
    
    expected_func = expected.get("function_name")
    expected_has_call = expected.get("requires_function", True)
    
    if not expected_has_call:
        # Negative sample - should NOT have a function call
        result["no_call_correct"] = not prediction["has_function_call"]
        return result
    
    # Positive sample - should have correct function call
    if prediction["has_function_call"] and prediction["function_name"]:
        result["function_correct"] = (
            prediction["function_name"].lower() == expected_func.lower()
        )
        
        # Check arguments (basic string match for now)
        if result["function_correct"] and prediction["arguments"]:
            pred_args = prediction["arguments"]
            exp_args = expected.get("arguments", {})
            
            if isinstance(pred_args, str):
                try:
                    pred_args = json.loads(pred_args)
                except:
                    pass
            if isinstance(exp_args, str):
                try:
                    exp_args = json.loads(exp_args)
                except:
                    pass
            
            # Simple check: at least one argument value matches
            if isinstance(pred_args, dict) and isinstance(exp_args, dict):
                for key, value in exp_args.items():
                    if key in pred_args and str(pred_args[key]) == str(value):
                        result["arguments_correct"] = True
                        break
    
    return result


def run_baseline_inference(
    model_name: str = "google/gemma-2b-it",
    dataset_name: str = "Sa74ll/arabic-mobile-actions",
    num_samples: int = 100,
    output_dir: str = "data/baseline",
    seed: int = 42
):
    """
    Run baseline inference on Arabic function calling dataset.
    
    Args:
        model_name: HuggingFace model name
        dataset_name: Dataset to test on
        num_samples: Number of samples to test
        output_dir: Directory to save results
        seed: Random seed
    """
    random.seed(seed)
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Load model
    model, tokenizer = load_model_and_tokenizer(model_name)
    
    # Load dataset
    console.print(f"\n[bold blue]Loading dataset: {dataset_name}[/bold blue]")
    dataset = load_dataset(dataset_name, split="train")
    
    # Sample subset
    if num_samples < len(dataset):
        indices = random.sample(range(len(dataset)), num_samples)
        samples = [dataset[i] for i in indices]
    else:
        samples = list(dataset)
    
    console.print(f"Testing on {len(samples)} samples\n")
    
    # Track results
    results = []
    metrics = {
        "total": 0,
        "positive_samples": 0,
        "negative_samples": 0,
        "function_correct": 0,
        "arguments_correct": 0,
        "no_call_correct": 0,
        "by_dialect": defaultdict(lambda: {"total": 0, "correct": 0}),
        "by_domain": defaultdict(lambda: {"total": 0, "correct": 0}),
        "errors": []
    }
    
    # Run inference
    for sample in tqdm(samples, desc="Running inference"):
        metrics["total"] += 1
        
        # Get sample info - handle both old and new format
        # New format: messages[0]=developer, messages[1]=user, messages[2]=assistant
        # Old format: messages[0]=user, messages[1]=assistant
        messages = sample["messages"]
        
        # Find user message (first message with role="user")
        user_query = None
        assistant_msg = {}
        for i, msg in enumerate(messages):
            if msg.get("role") == "user":
                user_query = msg["content"]
            elif msg.get("role") == "assistant":
                assistant_msg = msg
        
        if user_query is None:
            user_query = messages[0]["content"]  # Fallback
            
        tools = sample.get("tools", [])
        dialect = sample.get("dialect", "unknown")
        domain = sample.get("domain", "unknown")
        
        # Check if positive or negative sample
        has_tool_call = "tool_calls" in assistant_msg and assistant_msg["tool_calls"]
        
        expected = {
            "requires_function": has_tool_call,
            "function_name": None,
            "arguments": None
        }
        
        if has_tool_call:
            metrics["positive_samples"] += 1
            tool_call = assistant_msg["tool_calls"][0]
            expected["function_name"] = tool_call["function"]["name"]
            expected["arguments"] = tool_call["function"]["arguments"]
        else:
            metrics["negative_samples"] += 1
        
        # Generate prediction
        try:
            response = generate_response(model, tokenizer, user_query, tools)
            prediction = extract_function_call(response)
        except Exception as e:
            prediction = {
                "raw_response": f"ERROR: {str(e)}",
                "has_function_call": False,
                "parse_error": str(e)
            }
            metrics["errors"].append(str(e))
        
        # Evaluate
        eval_result = evaluate_prediction(prediction, expected)
        
        # Update metrics
        if has_tool_call:
            if eval_result["function_correct"]:
                metrics["function_correct"] += 1
                metrics["by_dialect"][dialect]["correct"] += 1
                metrics["by_domain"][domain]["correct"] += 1
            if eval_result["arguments_correct"]:
                metrics["arguments_correct"] += 1
        else:
            if eval_result["no_call_correct"]:
                metrics["no_call_correct"] += 1
        
        metrics["by_dialect"][dialect]["total"] += 1
        metrics["by_domain"][domain]["total"] += 1
        
        # Store result
        results.append({
            "query": user_query,
            "dialect": dialect,
            "domain": domain,
            "expected": expected,
            "prediction": prediction,
            "evaluation": eval_result
        })
    
    # Calculate final metrics
    final_metrics = {
        "model": model_name,
        "dataset": dataset_name,
        "num_samples": metrics["total"],
        "timestamp": datetime.now().isoformat(),
        "function_accuracy": metrics["function_correct"] / max(metrics["positive_samples"], 1),
        "argument_accuracy": metrics["arguments_correct"] / max(metrics["positive_samples"], 1),
        "negative_accuracy": metrics["no_call_correct"] / max(metrics["negative_samples"], 1),
        "positive_samples": metrics["positive_samples"],
        "negative_samples": metrics["negative_samples"],
        "by_dialect": {},
        "by_domain": {},
        "errors": metrics["errors"][:10]  # Keep first 10 errors
    }
    
    # Calculate per-dialect/domain accuracy
    for dialect, data in metrics["by_dialect"].items():
        if data["total"] > 0:
            final_metrics["by_dialect"][dialect] = {
                "total": data["total"],
                "correct": data["correct"],
                "accuracy": data["correct"] / data["total"]
            }
    
    for domain, data in metrics["by_domain"].items():
        if data["total"] > 0:
            final_metrics["by_domain"][domain] = {
                "total": data["total"],
                "correct": data["correct"],
                "accuracy": data["correct"] / data["total"]
            }
    
    # Print results
    console.print("\n" + "="*60)
    console.print("[bold green]BASELINE INFERENCE RESULTS[/bold green]")
    console.print("="*60)
    
    summary_table = Table(title="Summary")
    summary_table.add_column("Metric", style="cyan")
    summary_table.add_column("Value", style="green")
    
    summary_table.add_row("Model", model_name)
    summary_table.add_row("Total samples", str(metrics["total"]))
    summary_table.add_row("Positive samples", str(metrics["positive_samples"]))
    summary_table.add_row("Negative samples", str(metrics["negative_samples"]))
    summary_table.add_row("Function accuracy", f"{final_metrics['function_accuracy']:.1%}")
    summary_table.add_row("Argument accuracy", f"{final_metrics['argument_accuracy']:.1%}")
    summary_table.add_row("Negative accuracy", f"{final_metrics['negative_accuracy']:.1%}")
    
    console.print(summary_table)
    
    # Dialect breakdown
    dialect_table = Table(title="Accuracy by Dialect")
    dialect_table.add_column("Dialect", style="cyan")
    dialect_table.add_column("Samples", style="white")
    dialect_table.add_column("Accuracy", style="green")
    
    for dialect, data in sorted(final_metrics["by_dialect"].items(), key=lambda x: -x[1]["accuracy"]):
        dialect_table.add_row(dialect, str(data["total"]), f"{data['accuracy']:.1%}")
    
    console.print(dialect_table)
    
    # Domain breakdown
    domain_table = Table(title="Accuracy by Domain")
    domain_table.add_column("Domain", style="cyan")
    domain_table.add_column("Samples", style="white")
    domain_table.add_column("Accuracy", style="green")
    
    for domain, data in sorted(final_metrics["by_domain"].items(), key=lambda x: -x[1]["accuracy"]):
        domain_table.add_row(domain, str(data["total"]), f"{data['accuracy']:.1%}")
    
    console.print(domain_table)
    
    # Save results
    results_path = Path(output_dir) / "baseline_results.json"
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    metrics_path = Path(output_dir) / "baseline_metrics.json"
    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump(final_metrics, f, ensure_ascii=False, indent=2)
    
    console.print(f"\n[bold]Results saved to:[/bold]")
    console.print(f"  • {results_path}")
    console.print(f"  • {metrics_path}")
    
    return final_metrics


def main():
    parser = argparse.ArgumentParser(
        description="Run baseline inference for FunctionGemma on Arabic"
    )
    parser.add_argument(
        "--model", "-m",
        default="google/gemma-2b-it",
        help="Model name (default: google/gemma-2b-it)"
    )
    parser.add_argument(
        "--dataset", "-d",
        default="Sa74ll/arabic-mobile-actions",
        help="Dataset to test on"
    )
    parser.add_argument(
        "--samples", "-n",
        type=int,
        default=100,
        help="Number of samples to test"
    )
    parser.add_argument(
        "--output", "-o",
        default="data/baseline",
        help="Output directory"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    
    args = parser.parse_args()
    
    run_baseline_inference(
        model_name=args.model,
        dataset_name=args.dataset,
        num_samples=args.samples,
        output_dir=args.output,
        seed=args.seed
    )


if __name__ == "__main__":
    main()
