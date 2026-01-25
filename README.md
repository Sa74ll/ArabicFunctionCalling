# Arabic Function Calling with FunctionGemma

The first Arabic on-device function-calling system using Google's FunctionGemma (270M parameters).

##  Results

| Metric | Value |
|--------|-------|
| **Function Accuracy** | **71.9%** |
| Argument Accuracy | 43.2% |
| Negative Accuracy | 86.7% |
| Model Size | 270M parameters |

### Accuracy by Dialect

| Dialect | Accuracy |
|---------|----------|
| Gulf | 69.6% |
| MSA | 67.6% |
| Egyptian | 65.2% |
| Levantine | 62.5% |
| Maghrebi | 60.0% |

##  Dataset

**[Sa74ll/arabic-mobile-actions](https://huggingface.co/datasets/Sa74ll/arabic-mobile-actions)**

- 45,729 samples
- 5 Arabic dialects
- 31 functions across 8 domains
- Includes developer role for FunctionGemma compatibility

##  Quick Start

```bash
# Clone the repo
git clone https://github.com/Sa74ll/arabic-function-calling.git
cd arabic-function-calling

# Install dependencies
pip install torch transformers datasets

# Run inference
python inference.py
```

##  Usage

```python
from inference import load_model, call_function

# Load model
model, tokenizer = load_model()

# Define a tool
weather_tool = {
    "function": {
        "name": "get_weather",
        "description": "Get weather for a city",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string"}
            },
            "required": ["city"]
        }
    }
}

# Call function with Arabic query
result = call_function(
    model, tokenizer,
    query="ما حالة الطقس في الرياض؟",
    tools=[weather_tool]
)

print(result["function_name"])  # get_weather
print(result["arguments"])      # {"city": "الرياض"}
```

##  Project Structure

```
arabic-function-calling/
├── inference.py              # Simple inference script
├── src/
│   ├── convert_dataset.py    # Dataset conversion
│   ├── validate_dataset.py   # Validation
│   ├── baseline_inference.py # Full evaluation
│   └── upload_to_hub.py      # HuggingFace upload
├── schemas/
│   └── function_schemas.json # 31 function schemas
└── data/
    └── processed/            # Converted dataset
```

##  Key Discovery

> **FunctionGemma requires a `developer` role message to activate function calling mode.**

Without it: 0% accuracy. With it: 71.9% accuracy.

```python
messages = [
    {"role": "developer", "content": "You are a helpful assistant..."},
    {"role": "user", "content": "Arabic query..."}
]
```

##  Domains Covered

- Islamic Services (prayer times, Qibla, Zakat)
- Government Services (visa, ID, traffic fines)
- Banking/Finance (transfer, loans, gold prices)
- Healthcare (appointments, medications)
- Travel (flights, hotels)
- Weather
- E-commerce
- Utilities

##  Citation

```bibtex
@dataset{arabic_mobile_actions,
  author = {Sa74ll},
  title = {Arabic Mobile Actions Dataset},
  year = {2026},
  url = {https://huggingface.co/datasets/Sa74ll/arabic-mobile-actions}
}
```

## 📄 License

Apache 2.0
