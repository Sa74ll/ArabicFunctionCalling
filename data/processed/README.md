---
license: apache-2.0
language:
- ar
tags:
- function-calling
- tool-use
- arabic
- mobile-actions
- functiongemma
- on-device
size_categories:
- 10K<n<100K
task_categories:
- text-generation
pretty_name: Arabic Mobile Actions
source_datasets:
- HeshamHaroon/Arabic_Function_Calling
---

# Arabic Mobile Actions Dataset

**Arabic Function Calling dataset reformatted for FunctionGemma fine-tuning.**

This dataset converts the [Arabic Function Calling Dataset](https://huggingface.co/datasets/HeshamHaroon/Arabic_Function_Calling) to the [Google Mobile Actions](https://huggingface.co/datasets/google/mobile-actions) format, enabling fine-tuning of FunctionGemma for Arabic on-device function calling.

## Dataset Statistics

| Metric | Value |
|--------|-------|
| Total samples | 45,729 |
| Train samples | 36,583 (80%) |
| Eval samples | 9,146 (20%) |
| Positive (with tool call) | 41,175 |
| Negative (no tool call) | 4,554 |
| Functions | 31 |
| Domains | 8 |

### Dialect Distribution
| Dialect | Count | Percentage |
|---------|-------|------------|
| MSA (Modern Standard Arabic) | 14,181 | 31.0% |
| Gulf | 11,768 | 25.7% |
| Egyptian | 10,836 | 23.7% |
| Levantine | 6,711 | 14.7% |
| Maghrebi | 2,233 | 4.9% |

### Domains
- Government Services, Islamic Services, Healthcare, Banking/Finance
- Weather, Utilities, Travel, E-commerce

## Dataset Format

Each sample includes the **developer role** required for FunctionGemma:

```json
{
  "metadata": "train",
  "tools": [{
    "function": {
      "name": "get_prayer_times",
      "description": "Get prayer times / مواقيت الصلاة",
      "parameters": {...}
    }
  }],
  "messages": [
    {
      "role": "developer",
      "content": "You are a helpful assistant that can use tools..."
    },
    {
      "role": "user", 
      "content": "عايز أعرف مواعيد الصلاة في القاهرة"
    },
    {
      "role": "assistant",
      "tool_calls": [{
        "id": "call_xxx",
        "type": "function",
        "function": {
          "name": "get_prayer_times",
          "arguments": "{\"city\": \"القاهرة\"}"
        }
      }]
    }
  ],
  "dialect": "Egyptian",
  "domain": "islamic_services"
}
```

> **Important:** The `developer` role is required for FunctionGemma to activate function calling mode.
```

## Usage

```python
from datasets import load_dataset

# Load dataset
dataset = load_dataset("Sa74ll/arabic-mobile-actions")

# Filter by split
train_data = dataset.filter(lambda x: x["metadata"] == "train")
eval_data = dataset.filter(lambda x: x["metadata"] == "eval")

# Filter by dialect
egyptian = dataset.filter(lambda x: x.get("dialect") == "Egyptian")
```

## Fine-tuning with FunctionGemma

Compatible with [FunctionGemma Mobile Actions notebook](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/FunctionGemma_(270M)-Mobile-Actions.ipynb).

## Citation

If you use this dataset, please cite the original Arabic Function Calling dataset:

```bibtex
@dataset{arabic_function_calling,
  author = {Hesham Haroon},
  title = {Arabic Function Calling Dataset},
  year = {2024},
  url = {https://huggingface.co/datasets/HeshamHaroon/Arabic_Function_Calling}
}
```

## License

Apache 2.0
