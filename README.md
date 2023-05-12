## DEnsity: Open-domain Dialogue Evaluation Metric using Density Estimation

This repository contains the code and pre-trained models for [DEnsity: Open-domain Dialogue Evaluation Metric using Density Estimation (ACL2023 Findings)](https://arxiv.org/pdf/2305.04720.pdf).

### 0. Preparation

**1. Requirements**
```
torch==1.7.1
transformers==4.12.3
```

**2. Download pre-trained models**

Model checkpoints (trained on DailyDialog): [Link](TBA)
Statistic files of human responses (DailyDialog): [Link](TBA)

Locate the downloaded files as below:
```
DEnsity/
    logs/
        dd/
            reranker.scl-temp0.1-coeff1.epoch10.lr5e-5/
                models/
                    bestmodel.pth
    results/
        pickle_save_path/
            dd/
                maha.ref-train.reranker-reranker.scl-temp0.1-coeff1.epoch10.lr5e-5.positive.pck
```


### 1. How to use DEnsity for Evaluation?
```
from evaluators.model import DEnsity
from utils.utils import load_tokenizer_and_reranker


lm_name = 'bert-base-uncased'
model_path = './logs/dd/reranker.scl-temp0.1-coeff1.epoch10.lr5e-5/models/bestmodel.pth'
mean_cov_pck_fname = "./results/pickle_save_path/dd/maha.ref-train.reranker-reranker.scl-temp0.1-coeff1.epoch10.lr5e-5.positive.pck"

tokenizer, model = load_tokenizer_and_reranker(lm_name, model_path)
evaluator = DEnsity(None,mean_cov_pck_fname,tokenizer,model)

conversation = ["How are you?", "I'm fine, thank you!", "That's great!!!!"]

turn_level_score = evaluator.evaluate(conversation, is_turn_level=True)
dialogue_level_score = evaluator.evaluate(conversation, is_turn_level=False)
```

### 2. How to Train DEnsity from Scratch?
Below procedure is an example of training our feature extractor (i.e. response selection model) on DailyDialog dataset.

**1. Dataset Preparation**

Download [DailyDialog](https://aclanthology.org/I17-1099/) dataset and locate it as below.

```
DEnsity/
    data/
        dd/
            train/
                dialogues_train.txt
            validation/
                dialogues_validation.txt
```

**2. Run Training**
```
source scripts/train.sh
```

**3. How to train on new datasets other than DailyDialog?**
- Make a new dataset class (e.g., `MyDatasetforSelection(SelectionDataset)`).

### 3. How to Reproduce the Paper Result?

**1. Preprocessing evaluation dataset**
```
# DailyDialogue-Zhao
# Download human annotation file from [here](https://drive.google.com/drive/folders/1Y0Gzvxas3lukmTBdAI6cVC4qJ5QM0LBt) to `data/evaluation/dd/dd_annotations.json`.
python preprocess/preprocess_dd_zhao_annotation.py

# GRADE-DailyDialogue
# Download human annotation file fore [here](https://github.com/li3cmz/GRADE/tree/main/evaluation).
python preprocess/preprocess_grade_annotation.py

# Dialogue-level FED
python preprocess/preprocess_fed_dialogue.py
```

**2. Run Evaluation**
DailyDialog Main Results
```
eval_batch_size=128
ref_type=train
expname=reranker.scl-temp0.1-coeff1.epoch10.lr5e-5
distance_method=maha

train_dtaset=dd
for eval_dataset in  dd  grade_eval_dd
do
    python evaluators/run_evaluation.py --reranker_exp_name $expname --distance_function $distance_method --ref_type $ref_type --eval_dataset $eval_dataset --dataset $train_dtaset  --eval_batch_size $eval_batch_size
done

```

To reproduce the results of dialogue-level evaluation with FED datset, please use the python code.
```
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
