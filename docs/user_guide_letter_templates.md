# 📖 User Guide: Letter Template Creation

This guide explains how to create and customize **letter templates** — the character-level training sequences used by Letter-Forge's Transformer Encoder (Part 1).

---

## 📌 What Is a Letter Template?

A **letter template** is a single line of text, exactly **20 characters long**, used as a training or evaluation example for the Transformer Encoder.

Each line becomes one example that the model processes.  
The model learns to predict, for every character position, how many times that character has appeared earlier in the same line.

**Example line (exactly 20 characters including the trailing space):**

```
man many in the anar
```

Counted occurrences for each position (task: `BEFORE`):

```
Position:  m  a  n     m  a  n  y     i  n     t  h  e     a  n  a  r
Count:     0  0  0  0  1  1  1  0  1  0  2  1  0  0  0  2  2  3  3  0
```

---

## 📐 Format Requirements

| Property | Requirement |
|----------|-------------|
| **Line length** | Exactly 20 characters per line (not counting the newline) |
| **Character set** | Lowercase letters `a`–`z` and the space character only |
| **Encoding** | Plain UTF-8 text file, Unix line endings (`\n`) recommended |
| **No blank lines** | Every line must contain exactly 20 valid characters |

> ⚠️ **Important:** The vocabulary is fixed to 27 symbols — `a` through `z` plus space.  
> Uppercase letters, digits, punctuation, and special characters are **not supported** and will cause an index error at runtime.

---

## 🛠️ Step-by-Step: Creating a Custom Dataset

### Step 1 — Prepare your source text

Start from any plain-text source (prose, code comments, Wikipedia, etc.) and lowercase it:

```python
text = "The quick brown fox jumps over the lazy dog."
text = text.lower()
```

### Step 2 — Keep only valid characters

Strip every character that is not `a`–`z` or a space:

```python
import re
text = re.sub(r"[^a-z ]", "", text)
```

### Step 3 — Collapse multiple spaces

Normalize runs of spaces into a single space:

```python
text = re.sub(r" {2,}", " ", text)
```

### Step 4 — Slice into 20-character chunks

```python
lines = [text[i:i+20] for i in range(0, len(text) - 19, 20)]
# Keep only lines that are exactly 20 characters long
lines = [l for l in lines if len(l) == 20]
```

### Step 5 — Write the files

```python
import random, math

random.shuffle(lines)
split = math.ceil(len(lines) * 0.9)  # 90 / 10 train-dev split

with open("data/my-train.txt", "w") as f:
    f.write("\n".join(lines[:split]) + "\n")

with open("data/my-dev.txt", "w") as f:
    f.write("\n".join(lines[split:]) + "\n")
```

### Step 6 — Verify the files

Quick sanity check before training:

```python
with open("data/my-train.txt") as f:
    for i, line in enumerate(f, 1):
        line = line.rstrip("\n")
        assert len(line) == 20, f"Line {i} has length {len(line)}: {repr(line)}"
        assert all(c in "abcdefghijklmnopqrstuvwxyz " for c in line), \
            f"Line {i} contains invalid characters: {repr(line)}"
print("All lines OK.")
```

---

## 🚀 Using a Custom Dataset

Pass your new files via the `--train` and `--dev` flags when running the encoder:

```bash
cd part_1_encoder
python letter_counting.py --train data/my-train.txt --dev data/my-dev.txt
```

To change the **task mode**, use the `--task` flag:

```bash
# Count only characters that appeared BEFORE the current position (default)
python letter_counting.py --task BEFORE --train data/my-train.txt --dev data/my-dev.txt

# Count characters that appeared both BEFORE and AFTER the current position
python letter_counting.py --task BEFOREAFTER --train data/my-train.txt --dev data/my-dev.txt
```

---

## 🔡 Task Modes Explained

| Mode | Description | Output label at position _i_ |
|------|-------------|-------------------------------|
| `BEFORE` (default) | Count prior occurrences of the character at position _i_ | 0 = never seen, 1 = seen once, 2 = seen ≥ 2 times before |
| `BEFOREAFTER` | Count all **other** occurrences of the character anywhere in the line | 0 = unique, 1 = one other, 2 = two or more others |

Both modes produce labels in the set `{0, 1, 2}` for each of the 20 positions.

---

## ⚙️ Hyperparameter Overrides

The trainer supports environment variable overrides so you can experiment without editing source files:

| Variable | Default | Description |
|----------|---------|-------------|
| `LF_D_MODEL` | `64` | Embedding / model width |
| `LF_D_INTERNAL` | `64` | Attention key/query dimension |
| `LF_LAYERS` | `1` | Number of Transformer encoder layers |
| `LF_EPOCHS` | `5` | Training epochs |
| `LF_LR` | `1e-3` | Learning rate |

**Example:**

```bash
LF_D_MODEL=128 LF_LAYERS=2 LF_EPOCHS=8 \
    python letter_counting.py --task BEFORE
```

---

## ⚠️ Common Errors and How to Fix Them

### 1. `AssertionError: Expected unbatched [L], got ...`

**Cause:** A template line is not exactly 20 characters long — the model is configured for `num_positions = 20`.

**Fix:** Re-run the Step 6 verification script above and remove or pad any lines that are not exactly 20 characters.

---

### 2. `IndexError: index out of range` during training

**Cause:** A character in the file is outside the 27-symbol vocabulary (`a`–`z` + space). Common culprits are uppercase letters, digits, punctuation, or Unicode characters.

**Fix:** Re-run Step 2 and Step 3 of the preparation pipeline to strip invalid characters.

---

### 3. Very low training accuracy (< 50 %)

**Cause:** Insufficient data or an inappropriate hyperparameter configuration.

**Fix:**
- Ensure you have at least **1 000 training lines** (ideally 5 000+).
- Try increasing model capacity: `LF_D_MODEL=128 LF_LAYERS=2`.
- Increase epochs: `LF_EPOCHS=8`.
- Lower the learning rate slightly: `LF_LR=8e-4`.

---

### 4. `FileNotFoundError: data/lettercounting-train.txt`

**Cause:** The script is being run from a directory other than `part_1_encoder/`.

**Fix:** Always `cd` into the module directory before running, or supply explicit paths:

```bash
cd part_1_encoder
python letter_counting.py
# or from the project root:
python part_1_encoder/letter_counting.py \
    --train part_1_encoder/data/lettercounting-train.txt \
    --dev   part_1_encoder/data/lettercounting-dev.txt
```

---

### 5. Attention plots not generated

**Cause:** The `plots/` directory does not exist.

**Fix:** Create it before running:

```bash
mkdir -p part_1_encoder/plots
python part_1_encoder/letter_counting.py
```

---

## 📁 Related Files

| Path | Description |
|------|-------------|
| `data/lettercounting-train.txt` | Default training set (20-char lines) |
| `data/lettercounting-dev.txt` | Default dev/evaluation set |
| `part_1_encoder/letter_counting.py` | Driver script (reads data, trains, evaluates) |
| `part_1_encoder/transformer.py` | Encoder implementation (`LetterCountingExample`, `train_classifier`) |
| `part_1_encoder/utils.py` | `Indexer` helper for vocabulary mapping |
| `artifacts/` | Saved model checkpoints and run metadata JSON |
| `plots/` | Attention heatmap images |

---

## 🔗 See Also

- [README](../README.md) — Project overview, installation, and quick-start guide  
- [Part 2 Language Model](../part_2_lm/lm.py) — Transformer LM training on free-form text (no fixed line length required)
