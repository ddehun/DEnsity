import json
import os
from dataclasses import asdict, dataclass
from functools import partial
from typing import List

import numpy as np
import torch
from scipy.stats import pearsonr, spearmanr
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.dataset_util import selection_collate_fn


@dataclass
class EvaluationExample:
    history: List[str]
    answer: str
    response: str
    score: float
    modelname: str = None
    fact: List[str] = None


def save_prediction_output(output_fname: str, examples: List, predictions: List[float]):
    SAVEKEY = ["history", "answer", "response", "score", "modelname", "fact"]
    os.makedirs(os.path.dirname(output_fname), exist_ok=True)
    with open(output_fname, "w") as f:
        assert len(examples) == len(predictions)
        for i, e in enumerate(examples):
            e = {k: v for k, v in asdict(e).items() if k in SAVEKEY}
            e["pred_score"] = float(predictions[i])
            f.write(json.dumps(e) + "\n")


def get_conv_repr(args, dataset, settype: str, tokenizer, reranker) -> np.ndarray:
    num_candidate = args.num_negative + 1 if settype == "original" else 2  # response and answer

    partial_selection_collate_fn = partial(selection_collate_fn, pad_id=tokenizer.pad_token_id)
    loader = DataLoader(
        dataset,
        shuffle=False,
        batch_size=args.eval_batch_size,
        collate_fn=partial_selection_collate_fn,
        drop_last=False,
    )
    hidden_repr_list = []
    for idx, batch in enumerate(tqdm(loader)):
        ids, mask = (e.to("cuda", non_blocking=True) for e in batch)
        bs = int(len(ids) / num_candidate)

        # Ignore the negative from the second
        ids = ids.reshape(bs, num_candidate, -1)[:, 0]
        mask = mask.reshape(bs, num_candidate, -1)[:, 0]

        with torch.no_grad():
            _, hidden = reranker(ids, mask, return_hidden=True)

        hidden = hidden.reshape(bs, args.hidden_repr_dim).cpu().numpy()
        hidden_repr_list.append(hidden)

    reprs = np.concatenate(hidden_repr_list, 0)  # |dataset| X model_dim
    return reprs


def get_correlation(humanscores: List[float], modelscores: List[float]):
    assert len(humanscores) == len(modelscores)
    pearson = pearsonr(humanscores, modelscores)
    spearman = spearmanr(humanscores, modelscores)

    item = {
        "pearson-value": pearson[0],
        "pearson-p": pearson[1],
        "spearman-value": spearman[0],
        "spearman-p": spearman[1],
    }
    return item
