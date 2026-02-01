# 🏛️ The Master's Thesis: Arabic FunctionGemma Architectural Deep Dive
## A Comprehensive Study on Parameter Efficiency, Memory Optimization, and Gradient Dynamics

This guide provides a rigorous analysis of the fine-tuning process for **FunctionGemma-270M**. We bridge the gap between high-level code implementation and the underlying mathematical principles that allow a 270M model to achieve state-of-the-art performance in Arabic function calling.

---

### 𝚽 Section 1: Low-Rank Adaptation (LoRA) - The Linear Algebra of Tuning

In standard fine-tuning, we update the weight matrix $W \in \mathbb{R}^{d \times k}$. For Gemma-270M, the total parameter count makes full-rank updates computationally prohibitive. LoRA solves this by approximating the weight update $\Delta W$ through low-rank decomposition.

#### 1.1 The Fundamental Equation
The forward pass through a LoRA-enhanced linear layer is defined as:
$$h = W_0 x + \Delta W x = W_0 x + \frac{\alpha}{r} (A \times B) x$$

- **$W_0$ (Frozen Weights):** The pre-trained weights from Google.
- **$A \in \mathbb{R}^{d \times r}$:** The "In-Projection" matrix, initialized with Gaussian noise.
- **$B \in \mathbb{R}^{r \times k}$:** The "Out-Projection" matrix, initialized to zero.
- **$r$ (Rank):** The dimension of the low-rank space.

#### 1.2 The Code-to-Math Mapping
```python
model = FastModel.get_peft_model(
    model,
    r = 32,             # Our Rank (r)
    lora_alpha = 64,    # Scaling Factor (alpha)
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj"],
)
```
**Pedagogical Deep Dive:**
The scaling factor $\alpha/r$ (here $64/32 = 2$) is essentially the "learning intensity." Since we are training a tiny model (270M) on a complex new task (Arabic), we use a high scaling factor to ensure the new "adapters" exert significant influence over the pre-trained output. By targeting all projections (including the MLP's `gate_proj`), we allow the model to learn both the **Attention** (how tokens relate) and the **Knowledge** (how to map Arabic words to JSON values).

---

### 𝚽 Section 2: Memory Optimization via Selective Recomputation

The primary bottleneck in transformer training is the storage of activations during the forward pass to be used in the backward pass.

#### 2.1 The Memory Complexity Problem
In standard training, memory scales linearly with the number of layers:
$$\text{Memory}_{\text{std}} = O(L \cdot N \cdot D)$$
Where $L$ is layers, $N$ is sequence length, and $D$ is hidden dimension.

#### 2.2 The Triton Optimization
Unsloth utilizes custom **Triton Kernels** to implement **Selective Activation Recomputation**. Instead of storing the activation $h$, the system stores only the input and the random seed. During the backward pass, it re-executes the forward operation on-the-fly.

**Mathematical Result:**
The memory complexity is reduced to being independent of the number of layers:
$$\text{Memory}_{\text{unsloth}} = O(N \cdot D)$$

**The Code Implementation:**
```python
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/functiongemma-270m-it",
    use_gradient_checkpointing = "unsloth", # The Triton switch
)
```

---

### 𝚽 Section 3: Optimisation & Gradient Masking

We use **Supervised Fine-Tuning (SFT)** to minimize the **Cross-Entropy Loss** ($L$).

#### 3.1 The Loss Function
$$L = - \sum \log P(x_t | x_{<t})$$

#### 3.2 Label Masking for Precision
To prevent the model from wasting capacity learning the user's prompt, we apply a mask. 
**The Logic:**
We set labels for instruction tokens to $-100$. In the PyTorch implementation of Cross-Entropy, any label with value $-100$ is ignored in the sum.

**The Code Implementation:**
```python
trainer = train_on_responses_only(
    trainer,
    instruction_part = "<start_of_turn>user\n",
    response_part = "<start_of_turn>model\n",
)
```
**Academic Insight:** This forces the gradient $\nabla L$ to only reflect errors in the model's function-calling response. It ensures the model's weights are only updated to better understand "how to act," not "how to repeat the user."

---

### 𝚽 Section 4: The Convergence Schedule (Cosine Decay)

#### 4.1 The Learning Rate Dynamics
We use a **Warmup-then-Decay** strategy to navigate the loss landscape.
- **Warmup (200 steps):** Prevents the weights from diverging early by slowly increasing the LR.
- **Cosine Decay:** Smoothly reduces the LR to zero.

#### 4.2 The Decay Equation
$$\eta_t = \eta_{\text{min}} + \frac{1}{2}(\eta_{\text{max}} - \eta_{\text{min}})(1 + \cos(\frac{T_{\text{cur}}}{T_{\text{max}}}\pi))$$

**The Code Implementation:**
```python
args = SFTConfig(
    learning_rate = 1e-3,       # Peak Learning Rate
    lr_scheduler_type = "cosine",
    warmup_steps = 200,         # Ramp-up period
)
```
**Deep Dive:** The high peak LR ($10^{-3}$) is necessary because the LoRA rank $r=32$ represents a very narrow path in the total parameter space. We need a strong gradient to "push" the model into the specific manifold required for Arabic function calling.

---
