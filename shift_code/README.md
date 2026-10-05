# SHIFT: Stochastic Hidden-Trajectory Deflection for Removing Diffusion-based Watermark
## 📌 Overview
This repository implements **SHIFT**, a method for removing diffusion-based watermarks via stochastic trajectory deflection.

We evaluate SHIFT across multiple watermarking methods, including:
- Tree-Ring (TR)
- Gaussian Shading (GS)
- RingID (RI)
- ROBIN
- SFW
- WIND
- PRC
- GM
- SEAL

## 📌 Dependency

This project is built upon the official implementation of 
[MarkDiffusion](https://github.com/THU-BPM/MarkDiffusion).

Please first clone and set up the full MarkDiffusion repository:

```bash
git clone https://github.com/THU-BPM/MarkDiffusion.git
cd MarkDiffusion
```

## 📂 Setup

After cloning the MarkDiffusion repository, please organize the project structure as follows:

### 🔹 Step 1: Place our scripts into MarkDiffusion

Copy the `scripts/` directory from this repository into the root directory of `MarkDiffusion`:

MarkDiffusion/

├── scripts/ # ← our scripts (attack, detection, etc.)

├── config/

├── watermark/

├── ...

---



### 🔹 Step 2: Download models into MarkDiffusion

Please manually download the required models and place them in a `models/` directory at the same level as `MarkDiffusion`.

### 🔹 Required models

The following models are required:

- `sd2.1` (Stable Diffusion 2.1)
- `blip2-flan-t5-xl`
- `finetuned-caption-embedding`

---

### 🔹 Directory structure

After downloading, organize your directories as follows:

project_root/

├── MarkDiffusion/

├── models/

│ ├── sd2.1/

│ ├── blip2-flan-t5-xl/

│ ├── finetuned-caption-embedding/



### 🔹 Step 3: Install dependencies

pip install -r scripts/requirements.txt



## 🧪 Usage

### 🔹 Step 4: Generate watermarked images

Run the following script to generate images with diffusion-based watermarks:

```bash
NUM_IMAGES=100
python scripts/smoke_test_tr_gs_seal.py --num_images ${NUM_IMAGES}
```



### 🔹 Step 5: Run watermark removal attack (SHIFT)

Run the following script to perform watermark removal using SHIFT:


```bash
STRENGTHS=$(seq -s, 0 0.05 0.7)
MAX_IMAGES=100
PROMPT_INDEX="${PROMPT_INDEX:-0}"

python scripts/attack_stochastic_decoupling.py \
  --max_images ${MAX_IMAGES} \
  --strengths ${STRENGTHS} \
  --prompt_index ${PROMPT_INDEX}
```

### 🔹 Step 6: Detect watermarks and evaluate performance

#### (a) Per-image detection
Run the following script to detect watermarks for each attacked image:

```bash
ALGORITHMS="TR,GS,RI,ROBIN,SFW,GM,PRC,WIND,SEAL"
SEPARATED_OUTPUT_FILE="output/test_rui/detections_Separated.jsonl"
python scripts/detect_watermarks_Separated.py \
  --max_images ${MAX_IMAGES} \
  --output_file ${SEPARATED_OUTPUT_FILE} \
  --algorithms ${ALGORITHMS}
```

#### (b) Summary evaluation (ASR computation)
Run the following script to compute aggregated metrics (e.g., Attack Success Rate):

Example:
```bash
python scripts/detect_watermarks_Summary.py \
  --max_images 10 \
  --output_file output/summary.jsonl \
  --algorithms TR,GS,RI \
  --detections_file output/separated.jsonl
```

🔹 Output
- **Separated results (per image):**
output/separated.jsonl
- **Summary results (aggregated):**
output/summary.jsonl

