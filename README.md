
# HHS

## _Toward Automated Fine-Grained Hypothesis Discovery_

### Overview

HHS is designed to assist in generating finegrained research hypotheses in a **copilot setting**. Below is a step-by-step guide to configure and run the system.

---

### 🛠️ Step 0: Set Up the Python Environment

Create and activate a dedicated Conda environment, then install the required dependencies:

```bash
conda create -n hhs python=3.8
conda activate hhs
pip install -r requirements.txt
```

---

### 🔧 Step 1: Configure API Settings

Open `main.sh` and set the following:

- `api_key`
- `eval_api_key` (can be the same as `api_key`)
- `base_url`

---

### 🚀 Step 2: Run HHS

Then run:

```bash
bash main.sh
```

> ⚠️ This step may take about **1.5 hours per input bkg_id**.


