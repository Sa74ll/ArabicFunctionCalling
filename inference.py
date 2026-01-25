#!/usr/bin/env python3
"""
Simple Arabic Function Calling Inference

A minimal script to test FunctionGemma on Arabic queries.
"""

import json
import re
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_model(model_name: str = "google/functiongemma-270m-it"):
    """Load FunctionGemma model and tokenizer."""
    print(f"Loading {model_name}...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.bfloat16
    )
    
    print("✓ Model loaded!")
    return model, tokenizer


def call_function(
    model,
    tokenizer,
    query: str,
    tools: list,
    max_tokens: int = 256
) -> dict:
    """
    Call FunctionGemma with an Arabic query.
    
    Args:
        model: Loaded model
        tokenizer: Loaded tokenizer
        query: Arabic user query
        tools: List of available tools
        max_tokens: Max tokens to generate
    
    Returns:
        dict with function_name and arguments
    """
    # Build messages with developer role (required for FunctionGemma)
    messages = [
        {
            "role": "developer",
            "content": "You are a helpful assistant that can use tools to help the user. When the user asks for something that requires a tool, call the appropriate function."
        },
        {
            "role": "user",
            "content": query
        }
    ]
    
    # Apply chat template with tools
    prompt = tokenizer.apply_chat_template(
        messages,
        tools=tools,
        tokenize=False,
        add_generation_prompt=True
    )
    
    # Generate
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=0.1,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    response = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True
    ).strip()
    
    # Parse function call
    return parse_function_call(response)


def parse_function_call(response: str) -> dict:
    """Parse FunctionGemma output format."""
    result = {
        "raw_response": response,
        "function_name": None,
        "arguments": {}
    }
    
    # Pattern: <start_function_call>call function_name{args}<end_function_call>
    match = re.search(
        r'<start_function_call>call[:\s]+(\w+)\{(.*?)\}',
        response,
        re.DOTALL
    )
    
    if match:
        result["function_name"] = match.group(1)
        args_str = match.group(2)
        
        # Parse string arguments: key:<escape>value<escape>
        for key, value in re.findall(r'(\w+):<escape>(.*?)<escape>', args_str):
            result["arguments"][key] = value
        
        # Parse numeric arguments: key:number
        for key, value in re.findall(r'(\w+):(\d+(?:\.\d+)?)', args_str):
            if key not in result["arguments"]:
                result["arguments"][key] = value
    
    return result


# Example tools
WEATHER_TOOL = {
    "function": {
        "name": "get_weather",
        "description": "Get weather for a city",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "City name"},
                "days": {"type": "integer", "description": "Forecast days"}
            },
            "required": ["city"]
        }
    }
}

PRAYER_TOOL = {
    "function": {
        "name": "get_prayer_times",
        "description": "Get prayer times for a city",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "City name"}
            },
            "required": ["city"]
        }
    }
}


if __name__ == "__main__":
    # Load model
    model, tokenizer = load_model()
    
    # Example queries
    examples = [
        ("ما حالة الطقس في الرياض؟", [WEATHER_TOOL]),
        ("عايز أعرف مواعيد الصلاة في القاهرة", [PRAYER_TOOL]),
        ("كيف الجو في دبي لمدة 3 أيام؟", [WEATHER_TOOL]),
    ]
    
    print("\n" + "="*60)
    print("Arabic Function Calling Demo")
    print("="*60)
    
    for query, tools in examples:
        print(f"\n📝 Query: {query}")
        
        result = call_function(model, tokenizer, query, tools)
        
        if result["function_name"]:
            print(f"✅ Function: {result['function_name']}")
            print(f"   Arguments: {result['arguments']}")
        else:
            print(f"❌ No function call detected")
            print(f"   Response: {result['raw_response'][:100]}...")
