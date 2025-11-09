# ⚒️ Letter-Forge

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)
![PyTorch](https://img.shields.io/badge/Framework-PyTorch-red?logo=pytorch)
![Transformer](https://img.shields.io/badge/Model-Transformer-orange?logo=openai)
![Tasks](https://img.shields.io/badge/Tasks-Encoding%20%7C%20Language%20Modeling-yellow)
![License](https://img.shields.io/badge/License-MIT-green)
![Platform](https://img.shields.io/badge/Platform-Local%20%7C%20Offline-lightgrey)
![Status](https://img.shields.io/badge/Status-Active-brightgreen)
![Last Updated](https://img.shields.io/badge/Last%20Updated-November%202025-purple)
![Project Type](https://img.shields.io/badge/Project-From%20Scratch%20Implementation-brown)

### 🧠 Meaning of the Name
> *Letter-Forge* symbolizes both the ancient craft of shaping written language  
> and the modern act of building models that learn linguistic structure from scratch.

> *A forge where letters learn to think.*

**Letter-Forge** is a from-scratch implementation of Transformer architectures for character-level learning and language modeling.  
It explores how attention, memory, and positional structure can emerge from simple sequences of letters - transforming raw symbols into learned meaning.

---

## 🧠 Overview

Letter-Forge is built to **craft language understanding from the ground up**.  
It begins with a minimalist Transformer Encoder that learns counting and pattern recognition at the character level,  
and extends to a full Transformer Language Model capable of predicting and generating text sequences.

Each component - from self-attention to positional encoding - is implemented manually to illustrate the inner mechanics of modern deep learning models.

---

## 🏗️ Architecture Highlights

| Component | Description |
|------------|-------------|
| **Custom Transformer Encoder** | Built from first principles using PyTorch layers (`Linear`, `Softmax`, `ReLU`) — no off-the-shelf Transformer modules. |
| **Self-Attention Mechanism** | Implements single-head attention using learned queries, keys, and values. Visualizes attention maps between character positions. |
| **Positional Encoding** | Supports both learned and sinusoidal positional embeddings to inject order awareness into the model. |
| **Transformer Language Model (LM)** | Extends the encoder into a causal language model predicting the next character given context. |
| **Visualization & Analysis** | Generates heatmaps showing how the model “looks back” over previous symbols while learning structure. |

---

## 🧩 Project Structure

```
letter-forge/
│
├── part-1_encoder/              # Transformer Encoder (character-level)
│   ├── data/
│   │   ├── lettercounting-train.txt
│   │   └── lettercounting-dev.txt
│   ├── letter_counting.py       # Driver script
│   ├── transformer.py           # Core encoder + attention implementation
│   └── utils.py                 # Indexer & helper utilities
│
├── part-2_lm/                   # Transformer Language Model
│   ├── data/
│   │   ├── text8-100k.txt
│   │   ├── text8-dev.txt
│   │   └── text8-test.txt
│   ├── lm.py                    # Driver & evaluation (perplexity, sanity checks)
│   ├── transformer_lm.py        # LM model + training loop
│   └── utils.py
│
├── sandbox_utils/               # Development & testing scripts
│   ├── data_pipeline_verifier.py
│   ├── attention_pe_module_test.py
│   ├── attention_validation_suite.py
│   └── repro_training_logger.py
│
├── artifacts/                   # Saved models & metadata
├── plots/                       # Attention heatmaps & visual outputs
└── README.md
```

---

## ⚙️ Installation

```bash
# Clone the repository
git clone https://github.com/<your-username>/letter-forge.git
cd letter-forge

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate     # or .\venv\Scripts\activate on Windows

# Install dependencies
pip install torch numpy matplotlib
```

---

## 🚀 Usage

### 🧮 Part 1 – Transformer Encoder
Train and test the character-level counting model:

```bash
cd part-1_encoder
python letter_counting.py
```

- Predicts how many times each letter has appeared before in a sequence.  
- Visualizes attention maps showing how the model “looks back” over earlier tokens.

**Sample Attention Visualization:**

<img src="../plots/attn_sample_1.png" width="400"> <img src="../plots/attn_sample_2.png" width="400">

---

### 🔡 Part 2 – Transformer Language Model
Train a Transformer to predict the **next character** in a text sequence:

```bash
cd part-2_lm
python lm.py --model NEURAL
```

- Learns from the **text8** dataset (100k character subset).  
- Evaluates on perplexity and token-level likelihood.  
- Produces valid probability distributions for every step.

---

## 📊 Results Summary

| Model | Task | Metric | Result |
|--------|------|---------|--------|
| **Transformer Encoder** | Character counting (BEFORE) | Accuracy | 98.3 % |
| **Transformer Encoder** | Character counting (BEFOREAFTER) | Accuracy | 97–99 % (tuned) |
| **Transformer LM** | Next-char prediction (text8) | Perplexity | ≤ 7 (target) |
| **Attention Visualization** | Pattern detection | Highlights same-character attention clusters |

> The model successfully learns to identify prior occurrences of letters and extends this ability to generate context-aware text sequences.

---

## 🔥 Key Insights

- **Self-Attention = Contextual Memory:**  
  Each token attends to relevant predecessors, forming a learned memory of prior occurrences.  
- **Positional Encoding Enables Order Awareness:**  
  Without positional information, the model treats input as a bag of symbols; with it, order emerges.  
- **From Counting to Composition:**  
  The same underlying structure that counts characters can generate language — showing the continuum from perception to composition.

---

## 📁 Artifacts

- **Checkpoints:** `artifacts/model_*.pt`
- **Metadata:** `artifacts/run_meta.json`
- **Plots:** `plots/*.png` – attention heatmaps, loss curves, etc.

---

## 🧪 Sandbox Utilities

| Script | Purpose |
|---------|----------|
| `data_pipeline_verifier.py` | Validates dataset shapes and preprocessing pipeline. |
| `attention_pe_module_test.py` | Unit-tests Positional Encoding and Attention modules. |
| `attention_validation_suite.py` | Verifies attention masks and tensor consistency. |
| `repro_training_logger.py` | Reproducible 3-epoch experiment logger (saves artifacts). |

---

## 🪶 Philosophy

> *Letter-Forge* is built on a simple idea:  
> that language understanding isn’t magic - it’s forged through repeated interaction between memory, order, and meaning.  
> By crafting each layer manually, we can see how modern intelligence emerges, one letter at a time.

---

## 🧑‍💻 Author & Maintainer
**d-senyaka**   
AI & Data Science Developer · Deep Learning Enthusiast · Language Technology Researcher  

---

## ⚖️ License
This project is released under the **MIT License**.  
You are free to use, modify, and distribute it with attribution.

---

## ⭐ Acknowledgements
- Inspired by open research in attention mechanisms and neural sequence modeling.  
- Crafted with curiosity, patience, and an appreciation for both language and logic.

> *“To forge a mind of letters is to understand the art of attention.”*
