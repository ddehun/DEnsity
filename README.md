<!-- omit in toc -->
## DEnsity: Open-domain Dialogue Evaluation Metric using Density Estimation

This repository contains the code and pre-trained models for [DEnsity: Open-domain Dialogue Evaluation Metric using Density Estimation (ACL2023 Findings)](https://arxiv.org/pdf/2305.04720.pdf).


<!-- omit in toc -->
## Links
- [0. Preparation](#0-preparation)
- [1. How to use DEnsity for Evaluation?](#1-how-to-use-density-for-evaluation)
- [2. How to Train DEnsity from Scratch?](#2-how-to-train-density-from-scratch)
- [3. How to Reproduce the Paper Result?](#3-how-to-reproduce-the-paper-result)

### 0. Preparation

**1. Requirements**
```
torch==1.7.1
transformers==4.12.3
scikit-learn
scipy
wget
```

**2. Download pre-trained models**


Pretrained models (trained on DailyDialog and ConvAI2): [Link](https://drive.google.com/drive/folders/1IUUg6xsmEr28oed2yPqIA2m6xsQ9yNRd?usp=share_link)


Locate the downloaded files as below:
```bash
DEnsity/
    logs/
        dd/
            reranker.scl-temp0.1-coeff1.epoch10.lr5e-5/
                models/
                    bestmodel.pth
        convai2/
            reranker.scl-temp0.1-coeff1.epoch10.lr5e-5/
                models/
                    bestmodel.pth
    results/
        pickle_save_path/
            dd/
                maha.ref-train.reranker-reranker.scl-temp0.1-coeff1.epoch10.lr5e-5.positive.pck
            convai2/
                maha.ref-train.reranker-reranker.scl-temp0.1-coeff1.epoch10.lr5e-5.positive.pck
```

---

### 1. How to use DEnsity for Evaluation?
```python
from evaluators.model import DEnsity
from utils.utils import load_tokenizer_and_reranker


lm_name = 'bert-base-uncased'
model_path = './logs/dd/reranker.scl-temp0.1-coeff1.epoch10.lr5e-5/models/bestmodel.pth'
mean_cov_pck_fname = "./results/pickle_save_path/dd/maha.ref-train.reranker-reranker.scl-temp0.1-coeff1.epoch10.lr5e-5.positive.pck"

tokenizer, model = load_tokenizer_and_reranker(lm_name, model_path)
evaluator = DEnsity(None,mean_cov_pck_fname,tokenizer,model)

conversation = ["How are you?", "I'm fine, thank you!", "That's great!!!!"]

turn_level_score = evaluator.evaluate(conversation, is_turn_level=True) # -498.25882 
dialogue_level_score = evaluator.evaluate(conversation, is_turn_level=False) # -352.70813
```

---

### 2. How to Train DEnsity from Scratch?
Below procedure is an example of training our feature extractor (i.e. response selection model) on DailyDialog dataset.

**1. Dataset Preparation**

Download [DailyDialog](https://aclanthology.org/I17-1099/) dataset and locate it as below.

```bash
DEnsity/
    data/
        dd/
            train/
                dialogues_train.txt
            validation/
                dialogues_validation.txt
```

**2. Run Training**
```bash
source scripts/train.sh
```

---

**3. How to train on new datasets other than DailyDialog and ConvAi2?**

Make a new dataset class (e.g., `MyDatasetforSelection(SelectionDataset)`). You can refer to `ConvAI2forSelection()` or `ConvAI2forSelection()` classes in `utils/dataset_util.py`.


### 3. How to Reproduce the Paper Result?

**1. Preprocessing evaluation dataset**
```bash
# DailyDialogue-Zhao
# Download human annotation file from [here](https://drive.google.com/drive/folders/1Y0Gzvxas3lukmTBdAI6cVC4qJ5QM0LBt) to `data/evaluation/dd/dd_annotations.json`.
python preprocess/preprocess_dd_zhao_annotation.py

# GRADE-DailyDialog and GRADE-ConvAI2
# Download human annotation file fore [here](https://github.com/li3cmz/GRADE/tree/main/evaluation).
python preprocess/preprocess_grade_annotation.py

# USR-ConvAI2
python preprocess/preprocess_usr_annotation.py

# Dialogue-level FED
preprocess/preprocess_fed_dialogue.ipynb # Run notebook
```

**2. Run Evaluation**

Main Results (Turn-level evalatuion in Table 1 of the paper)
```bash
source scripts/test.sh
```

To reproduce the results of dialogue-level evaluation with FED datset, please use the python code.

```Python
from tqdm import tqdm

from evaluators.model import DEnsity
from evaluators.eval_utils import get_correlation
from utils.utils import load_tokenizer_and_reranker, read_jsonl


# Load model
lm_name = "bert-base-uncased"
model_path = "./logs/dd/reranker.scl-temp0.1-coeff1.epoch10.lr5e-5/models/bestmodel.pth"
mean_cov_pck_fname = "./results/pickle_save_path/dd/maha.ref-train.reranker-reranker.scl-temp0.1-coeff1.epoch10.lr5e-5.positive.pck"

tokenizer, model = load_tokenizer_and_reranker(lm_name, model_path)
evaluator = DEnsity(None, mean_cov_pck_fname, tokenizer, model)

# Read datset
fed_fname = "./data/evaluation/fed_dialogue_overall/test_processed.json"
fed_examples = read_jsonl(fed_fname)

# Run evaluation
model_scores = []
human_scores = []
tokenizer.truncation_side = "left"

for _, el in enumerate(tqdm(fed_examples)):
    uttrs, score = el["history"], el["score"]
    density_score = evaluator.evaluate(uttrs, is_turn_level=False)
    model_scores.append(density_score)
    human_scores.append(score)


print(round(100 * get_correlation(human_scores, model_scores)["spearman-value"], 2))  # 43.34
```