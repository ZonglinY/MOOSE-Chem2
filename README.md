
# MOOSE-Chem2: Exploring LLM Limits in Fine-Grained Scientific Hypothesis Discovery via Hierarchical Search

[![Twitter](https://img.shields.io/twitter/url/https/twitter.com/cloudposse.svg?style=social&label=Follow%20%40Us)](https://x.com/Yang_zy223)
[![arXiv](https://img.shields.io/badge/arXiv-b31b1b.svg)](https://arxiv.org/abs/2505.19209)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

MOOSE-Chem2 introduces the task of *fine-grained scientific hypothesis discovery*—enabling LLMs to move beyond vague ideas toward **detailed hypotheses with sufficient methodological specifics for lab testing**.

This repository includes the dataset, method implementation, and analysis scripts for MOOSE-Chem2. It also can be used in a copilot setting where it takes in customized research question.

---

### 🛠️ Step 0: Set Up the Python Environment

Create and activate a dedicated Conda environment, then install the required dependencies:

```bash
conda create -n msc python=3.10
conda activate msc
pip install -r requirements.txt
```

---

### 🔧 Step 1: Configure API Settings

Open `main.sh` and set the following:

- `api_type` (0: OpenAI; 1: Azure; 2: Gemini)
- `api_key`
- `eval_api_key` (can be the same as `api_key`)
- `base_url`

---

### 📋 Step 2: (Optional) Provide Custom Research Inputs — or Use the Default Benchmark

Set `function_mode=5` in `main.sh`.

Choose **one** of the following two paths:

#### Option A: Using Output from MOOSE-Chem

1. Set `--if_load_from_moosechem_ranking_file 1`
2. Set `--moosechem_ranking_file_path` to the `--output_dir` from `./Method/evaluate.py` in [MOOSE-Chem](https://github.com/ZonglinY/MOOSE-Chem)
3. Open `./Preprocessing/custom_research_background_dumping.py`, go to `moosechem_ranking_file_to_json()` and **manually fill in** `background_survey`.

#### Option B: Provide Your Own Research Direction

1. Set `--if_load_from_moosechem_ranking_file 0`
2. In `./Preprocessing/custom_research_background_dumping.py`, go to `research_background_to_json()` and manually set:
   - `research_question`
   - `background_survey`
   - `coarse_grained_hypothesis`

Then run:

```bash
bash main.sh
```

---

### ✏️ Step 3: Adjust Discipline-Specific Prompts

#### Option A: Use Built-in Prompts for Chemistry or Geophysics

If your research question falls within **chemistry** or **geophysics**, go to `./Method/utils.py` and import the appropriate prompt module:

```python
from Method.discipline_specific_prompt_chemistry load xxx  # or
from Method.discipline_specific_prompt_geophysics load xxx
```

Keep only the one relevant to your domain and **comment out** the other.

#### Option B: Create Custom Prompt for Another Discipline

If your research area is outside chemistry or geophysics:

1. Create a new prompt file by mimicking the structure of `discipline_specific_prompt_chemistry.py`
2. In `./Method/utils.py`, **comment out** the existing chemistry and geophysics imports
3. Import your custom prompt module instead

This ensures the system uses discipline-specific guidance tailored to your research field.

---

### 🚀 Step 4: Run MOOSE-Chem2

Set the following in `main.sh`:

- `function_mode=0`
- `model_name` and `eval_model_name` (usually the same)
- `if_hierarchical=1`
- `num_hierarchy=5`
- `if_use_custom_research_background_and_coarse_hyp=1` (if use customized input, else set to 0)
- `bkg_id` – index of the research input you created in Step 2, default is 0 (the first one)

Then run:

```bash
bash main.sh
```

> ⚠️ This step may take up to **400 minutes per input bkg_id**.

---

### 📊 Step 5: Display Results

To display output, set in `main.sh`:

- `function_mode=6`
- `--start_bkg_id` and `--end_bkg_id` to cover the range of `bkg_id`s you processed  
  (e.g., if only `bkg_id=0` was run, set both to 0)

If file loading errors occur:

- Open `./Preprocessing/display_hypothesis.py`
- Under `# used for match hypothesis file name`, ensure the parameter settings match those in Step 4 (`main.sh`, under `if [[ ${function_mode} -eq 0 ]]; then`)

Then run:

```bash
bash main.sh
```

---

## Display Tables Presented in the Paper

1. Download the checkpoint files from [this link](https://drive.google.com/file/d/10UMrh_UZD3uam0wYeasyXj9dPmeGrBip/view?usp=sharing).
2. Unzip the downloaded archive into the current directory.

### Table 1
- Change the `display_type` in `./Evaluation/display_result.py` to 1, then run `python ./Evaluation/display_result.py`

### Table 2
- Change the `display_type` in `./Evaluation/display_result.py` to 2, then run `python ./Evaluation/display_result.py`

### Table 3
- Change the `display_type` in `./Evaluation/display_result.py` to 3, then run `python ./Evaluation/display_result.py`

Now a part of Table 3 is shown. Other parts of Table 3 can be shown by adjusting:
- `pairwise_eval_model_name, pairwise_if_multiple_llm`
- `model_name_1, eval_model_name_1, if_multiple_llm_1`
- `model_name_2, eval_model_name_2, if_multiple_llm_2`


### Table 4

1. Open `./Evaluation/display_result.py` and set:

   ```python
   display_type = 3
   ```
2. Comment out the line:

   ```python
   start_end_id_pairs = [[24, 35]]
   ```

   and uncomment the line:

   ```python
   start_end_id_pairs = [[0, 4], [5, 13], [14, 23], [24, 35], [36, 50]]
   ```
3. Set the following variables:

   ```python
   pairwise_eval_model_name, pairwise_if_multiple_llm = ["gpt-4o-mini", 1]
   model_name_1, eval_model_name_1, if_multiple_llm_1 = ["gpt-4o-mini", "gpt-4o-mini", 0]
   model_name_2, eval_model_name_2, if_multiple_llm_2 = ["gpt-4o-mini", "gpt-4o-mini", 1]
   ```
4. Run the script:

   ```bash
   python ./Evaluation/display_result.py
   ```

### Table 5

1. Open `./Evaluation/display_result.py` and set `display_type = 2`.
2. Comment out the following lines:

   ```python
   if_multiple_llm = 1
   which_exp = [1, 1, 1]
   ```
3. Uncomment the following lines:

   ```python
   if_multiple_llm = 0
   which_exp = [1, 0, 0]
   ```
4. Run the script:

   ```bash
   python ./Evaluation/display_result.py
   ```


--- 

## Conduct Pairwise Evaluation and Calculate Recall on Existing Checkpoints

1. Open `main.sh` and set the following:

   * `function_mode=3`
   * Provide values for `api_type`, `api_key`, and `base_url`.

2. Fill in the parameters located between:

   ```bash
   # shared parameters for pairwise compare and f1 score compare (start)
   ...
   # shared parameters for pairwise compare and f1 score compare (end)
   ```

   * Set `pairwise_or_f1_compare_mode` to:

     * `1` to conduct **pairwise evaluation** between two local maximum hypotheses.
     * `2` to compare a local maximum hypothesis with the ground truth hypothesis and **calculate recall**.

3. If `pairwise_or_f1_compare_mode=1`:

   * Fill in the section between:

     ```bash
     # shared parameters for pairwise compare (start)
     ...
     # shared parameters for pairwise compare (end)
     ```
   * If `compare_mode=2`, also fill in the parameters between:

     ```bash
     # parameters for pairwise compare (compare_mode 2) (start)
     ...
     # parameters for pairwise compare (compare_mode 2) (end)
     ```

4. If `pairwise_or_f1_compare_mode=2`:

   * Fill in the parameters between:

     ```bash
     # parameters for f1 score compare (start)
     ...
     # parameters for f1 score compare (end)
     ```

5. Finally, run:

   ```bash
   bash main.sh
   ```

---

## Bib Info
If you found this repository useful, please consider 📑citing:

	@inproceedings{yang2025moose2,
      title     = {MOOSE-Chem2: Exploring LLM Limits in Fine-Grained Scientific Hypothesis Discovery via Hierarchical Search},
      author    = {Yang, Zonglin and Liu, Wanhao and Gao, Ben and Liu, Yujie and Li, Wei and Xie, Tong and Bing, Lidong and Ouyang, Wanli and Cambria, Erik and Zhou, Dongzhan},
      booktitle = {Advances in Neural Information Processing Systems},
      year      = {2025}
   }


---

## License

This project is licensed under the MIT License. See the [LICENSE](./LICENSE) file for details.




