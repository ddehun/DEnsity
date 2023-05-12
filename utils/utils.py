import json
import os
import random

import numpy as np
import torch
from transformers import AutoTokenizer

from reranker.models import DefaultSelectionModel

TURN_TOKEN = "[SEPT]"


def load_tokenizer_and_reranker(lmtype: str, ckpt_path: str):
    tokenizer = AutoTokenizer.from_pretrained(lmtype)
    tokenizer.add_special_tokens({"additional_special_tokens": [TURN_TOKEN]})
    reranker = DefaultSelectionModel(lmtype).to("cuda").eval()
    reranker.model.resize_token_embeddings(len(tokenizer))
    reranker.load_state_dict(torch.load(ckpt_path))
    return tokenizer, reranker


def write_txt(fname, data):
    with open(fname, "w") as f:
        f.write("\n".join([e.strip() for e in data]))


def read_txt(fname):
    with open(fname, "r") as f:
        return [l.strip() for l in f.readlines()]


def read_npy(fname):
    with open(fname, "rb") as f:
        return np.load(f)


def write_npy(data, fname):
    os.makedirs(os.path.dirname(fname), exist_ok=True)
    with open(fname, "wb") as f:
        np.save(f, data)


def read_jsonl(fname):
    with open(fname, "r") as f:
        return [json.loads(e) for e in f.readlines()]


def write_jsonl(fname, data):
    with open(fname, "w") as f:
        for el in data:
            f.write(json.dumps(el) + "\n")


def set_seed(random_seed: int = 42) -> None:
    """Set RNG seeds for python's `random` module, numpy and torch."""
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)


def setup_path(path: str, allow_exist_ok: bool = False):
    # allow_exist_ok for debugging
    os.makedirs(path, exist_ok=allow_exist_ok)
    os.makedirs(os.path.join(path, "models"), exist_ok=True)
    os.makedirs(os.path.join(path, "models/ctx"), exist_ok=True)
    os.makedirs(os.path.join(path, "models/hyp"), exist_ok=True)
    os.makedirs(os.path.join(path, "board"), exist_ok=True)
    os.makedirs(os.path.join(path, "eval"), exist_ok=True)
    os.makedirs(os.path.join(path, "data/tensors"), exist_ok=True)


def dump_config(args):
    with open(os.path.join(args.exp_path, "train_config.json"), "w") as f:
        json.dump(vars(args), f)


def get_mrr(logits, do_average=True):
    order = (-logits).argsort()
    rank = order.argsort()[:, 0].squeeze()
    mrrs = 1 / (rank + 1)
    if do_average:
        return sum(mrrs) / len(mrrs)
    else:
        return mrrs


def get_recall1_for_reranker(logits, do_average=True):
    order = (-logits).argsort()
    rank = order.argsort()[:, 0].squeeze()
    hit = [e == 0 for e in rank]
    if do_average:
        return sum(hit) / len(hit)
    else:
        return hit
