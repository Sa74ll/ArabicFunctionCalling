# 🦸‍♂️ The Architect's Handbook: From Math to Code (Hero Edition)
## Arabic FunctionGemma-270M Fine-Tuning Deep Dive

This guide maps the mathematical theory directly to the code snippets you are using in your training script.

---

### 🏛️ 1. Parameter Efficiency: LoRA Linear Algebra
**The Math:** $\Delta W = A \times B$
**The Snippet:**
```python
model = FastModel.get_peft_model(
    model,
    r = 32,             # The Rank (Bottleneck dimension)
    lora_alpha = 64,    # Scaling Factor (2x r)
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj"],
)
```
**Deep Dive:**
- **Rank ($r$):** Matrix $A$ is $(d \times 32)$ and $B$ is $(32 \times k)$. Instead of updating millions of weights, we update the product of these two small matrices.
- **Target Modules:** By targeting `gate`, `up`, and `down` (the MLP block), we ensure the model learns not just *how* to call a function (Attention), but *what* logic to use in the function parameters (MLP).

---

### 🚀 2. Memory Surgery: The Triton Kernels
**The Math:** Recomputation of activations to save $O(N^2)$ VRAM.
**The Snippet:**
```python
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/functiongemma-270m-it",
    load_in_4bit = True, # 4-bit NormalFloat (NF4)
    use_gradient_checkpointing = "unsloth", 
)
```
**Deep Dive:**
- **`load_in_4bit`:** Uses **Double Quantization**. It quantizes the weights to 4 bits, then quantizes the quantization constants. Math: $W_{q} = \text{round}(W / S)$, where $S$ is the scale factor.
- **`use_gradient_checkpointing = "unsloth"`:** This triggers a custom **Triton kernel** that manually executes the Chain Rule during the backward pass. It discards the activation $H = \text{SiLU}(x \times W)$ and recomputes it when needed, saving ~30% VRAM.

---

### 🎯 3. Optimization: Training on Responses Only
**The Math:** $\mathcal{L} = -\sum y \log(\hat{y})$ where labels are masked.
**The Snippet:**
```python
from unsloth.chat_templates import train_on_responses_only
trainer = train_on_responses_only(
    trainer,
    instruction_part = "<start_of_turn>user\n",
    response_part = "<start_of_turn>model\n",
)
```
**Deep Dive:**
- **The Masking:** Inside this function, all tokens between `<start_of_turn>user` and `<start_of_turn>model` have their ID set to **-100**.
- **The Result:** The Cross-Entropy loss ignores `-100`. The model is *never* penalized for getting the user's prompt wrong—it only "feels the pain" (loss) if it messes up the Arabic function call. This is critical for stabilizing the 270M model.

---

### ⚙️ 4. The Schedule: Cosine Decay
**The Math:** $\eta_t = \frac{1}{2} \eta_{max} (1 + \cos(\frac{t \pi}{T}))$
**The Snippet:**
```python
args = SFTConfig(
    learning_rate = 1e-3,
    lr_scheduler_type = "cosine",
    warmup_steps = 200,
)
```
**Deep Dive:**
- **`learning_rate = 1e-3`:** High LR is needed because $r=32$ is a relatively large update for a small model. We need a strong gradient signal to align Arabic text to English JSON keys.
- **`warmup_steps = 200`:** Prevents the "Zero-Shot" shock. It slowly ramps up the learning rate from 0 to 1e-3 over the first 200 steps to prevent weight explosion.

---

### 🛠 5. Final Checklist for Tuesday
1.  **Check WandB:** If `train/loss` is flat at 0.001, you are overfitting. Lower LR to `5e-4`.
2.  **Verify Output:** Run the inference script. If it outputs `{"action": "..."}` but the content is wrong, you need more data (Rank is fine).
3.  **Export:** Use `model.save_pretrained_gguf(..., quantization_method = "Q8_0")`.

---
