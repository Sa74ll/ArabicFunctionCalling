# Next Phase: Arabic FunctionGemma Fine-Tuning Prep
## From Zero to Hero: The Architectural Deep Dive

This folder contains the essential knowledge required to understand the fine-tuning process for **FunctionGemma-270M** using the **Unsloth** framework on Arabic datasets.

---

### 🏛️ 1. The Transformer Foundation (Gemma-270M)
Gemma-270M is a lightweight, decoder-only transformer.
*   **The Math:** It uses **RMSNorm** for stability and **RoPE (Rotary Positional Embeddings)** to manage token relationships.
*   **The Logic:** Because it only has 270M parameters, the "information density" is extremely high. Unlike a 7B model, every single parameter counts. Fine-tuning on a new language (Arabic) and a strict format (JSON/Function Calling) requires surgical precision.

### 🚀 2. The Unsloth Engine (Triton & Memory)
Unsloth isn't just a library; it's a performance optimization layer.
*   **Triton Kernels:** Unsloth replaces standard PyTorch functions with custom-written Triton kernels. 
    *   *Math:* It manually optimizes the **Backward Pass**. Instead of storing massive activation tensors, it recomputes them using optimized math, reducing VRAM usage by up to 70%.
*   **Gradient Checkpointing:** Specifically tuned for Gemma, allowing a context window of **4096 tokens** even on a free T4 GPU (16GB VRAM).

### 🧬 3. LoRA: Low-Rank Adaptation
We don't train the whole model; we train "adapters."
*   **The Math:** $W_{updated} = W_{base} + \frac{\alpha}{r}(A \times B)$
*   **Rank (r=32):** This defines the dimensionality of the update. For a tiny model, 32 is a "thick" rank, giving the model enough capacity to learn the complex RTL (Arabic) to LTR (JSON) mapping.
*   **Alpha (64):** The scaling factor. Setting this to $2 \times r$ is the industry standard for stable learning.

### 🎯 4. Training Strategy: "Response-Only"
*   **The Logic:** If the model predicts the user's prompt, it wastes gradients.
*   **The Math:** We use a mask (setting labels to `-100`) for the instruction part. The **Cross-Entropy Loss** function ignores these, forcing the model to only care about getting the **Function Call** correct.

### ⚙️ 5. Hyperparameter Breakdown
*   **Learning Rate (1e-3):** Very high. Necessary for tiny models to "forget" enough of their old weights to adopt the new Arabic function-calling patterns.
*   **Warmup Steps (200):** Crucial. It gradually increases the LR so the model doesn't "explode" when it first encounters the Arabic dataset.
*   **Epochs (2-3):** Small models overfit quickly. Keep it under 3 epochs and monitor the evaluation loss.

---

## 🛠 Next Steps (Before You Run 'Run All')
1.  **Check Tokenizer:** Ensure the Arabic text isn't being broken into too many sub-tokens.
2.  **Validate JSON:** Ensure your dataset's Mobile Action format is perfectly valid JSON.
3.  **On-Device Prep:** Plan to export as **GGUF (Q8_0)** for the best balance of speed and logic.
