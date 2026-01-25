# 🎓 Code Walkthrough: The Logic of Inference

**Target Audience:** Computer Science Students & Beginning AI Engineers.
**Subject:** Understanding how to run and evaluate an LLM programmatically.
**File:** `src/baseline_inference.py`

This document breaks down the code function-by-function. Instead of just showing you *what* the code does, we will explore *why* it was written that way.

---

## 1. `load_model_and_tokenizer`

### The Concept
Before we can "think", we must load the brain. This function handles the heavy lifting of moving the 270 million parameters from your hard drive into the GPU's VRAM.

### Key Logic
```python
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype=torch.bfloat16 if DEVICE == "cuda" else torch.float32,
)
```

**Student Note:**
*   **`device_map="auto"`**: This is a lifesaver. It automatically figures out if you have a GPU, how much memory it has, and where to put the model layers. Without this, you'd have to manually move tensors like `model.to("cuda")`.
*   **`torch.bfloat16`**: "Brain Floating Point". It uses fewer bits (16) than standard numbers (32), cutting memory usage in half. Unlike standard `float16`, it has the same "dynamic range" as `float32`, making it much more stable for AI math.

---

## 2. `clean_tools_for_functiongemma`

### The Concept
LLMs are picky. If you format a tool definition slightly wrong, the model might ignore it. This function acts as a **data sanitizer**.

### What it does:
1.  **Wraps in "type":** Some datasets just give you the function definition. FunctionGemma explicitly wants it wrapped: `{"type": "function", "function": {...}}`.
2.  **Removes Nulls:** If a parameter has no description, normal Python might leave it as `None`. We remove these keys because `None` (or `null` in JSON) confuses the tokenizer.

---

## 3. `generate_response` (The Core)

### The Concept
This is the "forward pass"—where the input text is transformed into output text.

### The "Developer" Role Trick
This is the most important part of the entire project.

```python
messages = [
    {
        "role": "developer",
        "content": "You are a helpful assistant that can use tools..."
    },
    ...
]
```

**Why "developer" and not "system"?**
Most tutorials tell you to use the `system` role to give instructions. However, FunctionGemma was fine-tuned (trained) to specifically pay attention to the `developer` role for tool-use instructions.
*   **Without this:** The model thinks you are just chatting.
*   **With this:** The model enters "Function Calling Mode".

### The Chat Template
```python
prompt = tokenizer.apply_chat_template(
    messages,
    tools=cleaned_tools,
    add_generation_prompt=True
)
```
**Student Note:** Never manually combine strings like `"<start_of_turn>user..."`. You will get it wrong. The `tokenizer.apply_chat_template` function looks at the model's internal configuration matches the exact formatting used during training.

---

## 4. `extract_function_call`

### The Concept
The model outputs text. We need a Python object. This function is a **Parser**. It uses Regular Expressions (Regex) to hunt for structure in the chaos of generated text.

### The Patterns
We look for four different "shapes" of answers, just in case the model hallucinates:

1.  **The Perfect Match (FunctionGemma Native):**
    ```regex
    <start_function_call>call:(\w+)\{(.*?)\}<end_function_call>
    ```
    This is what the model *should* output. Usage of `<>` tags makes it easy to find.

2.  **The "Close Enough":**
    Sometimes the model forgets the closing tag. We use a regex that looks for the start but ignores the end.

3.  **The "JSON Hallucination":**
    Sometimes models revert to standard JSON: `{"name": "get_weather"}`. We catch this too using `re.search(r'\{...\}')`.

### Argument Parsing (The Hard Part)
FunctionGemma saves arguments strangely:
`city:<escape>Cairo<escape>`

Standard JSON parsers (like `json.loads`) will crash on this. We wrote a custom sub-parser:
```python
# Look for: key:<escape>value<escape>
re.findall(r'(\w+):<escape>(.*?)<escape>', args_str)
```

---

## 5. `evaluate_prediction`

### The Concept
How do we know if the model is "right"? Using simple string matching (`==`) isn't enough because arguments can be in different orders.

### The Logic
1.  **Function Name Check:** Case-insensitive string match.
    *   `get_weather` == `Get_Weather` (True)
    
2.  **Argument Check:**
    *   **Goal:** "Did the model get the right city?"
    *   **Method:** We convert everything back to Python dictionaries and check if the key values match.
    *   *Why?* Because `{"a": 1, "b": 2}` is the same as `{"b": 2, "a": 1}`, but as strings `"a=1,b=2"` != `"b=2,a=1"`. Comparing Objects > Comparing Strings.

---

## 6. `run_baseline_inference`

### The Concept
This is the **Experiment Runner**. It loops through the dataset, runs the functions above, and acts as a scientist taking notes.

### Metrics Collected
*   **Total Samples (N):** Sample size matters. (We used N=200).
*   **Positive Accuracy:** How often did it call a tool when it should have?
*   **Negative Accuracy:** How often did it stays silent when it should have? (Crucial for a chatty assistant!)
*   **Domain Breakdown:** We categorize success by topic (Weather vs. Banking) to see where the model is "smart" or "dumb".

### Final Output
It saves two files:
1.  `_results.json`: Every single input, output, and score. (For debugging specific errors).
2.  `_metrics.json`: The high-level summary. (For your presentation slides).
