import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json
import numpy as np
import os
from dataclasses import asdict

import wget

from evaluators.eval_utils import EvaluationExample


def download_usr_dataset(output_path="."):
    dataset = {
        "test": "http://shikib.com/pc_usr_data.json",
        "valid": "http://shikib.com/pc_usr_data.json",
    }
    wget.download(dataset["test"], output_path)
    wget.download(dataset["valid"], output_path)


def _write_items_to_jsonl(data, fname):
    with open(fname, "w") as f:
        for el in data:
            f.write(json.dumps(el) + "\n")


def preprocess_usr_dataset(org_fname, output_fname):
    """Preprocessing DailyDialog++ dataset"""
    with open(org_fname, "r") as f:
        data = json.load(f)
    output_data = []

    for item in data:
        fact = [e.strip() for e in item["fact"].split("\n") if e != ""]
        context = [e for e in item["context"].split("\n") if e != ""]
        responses = item["responses"]
        reference = [e for e in responses if e["model"] == "Original Ground Truth"][0]["response"].strip()
        for response in responses:
            text = response["response"].strip()
            author = response["model"]
            if author == "Original Ground Truth":
                continue
            scores = response["Overall"]
            assert all([1 <= e <= 5 for e in scores])
            score = float(np.mean([(score - 1) / 4 for score in scores]))

            output_data.append(EvaluationExample(context, reference, text, score, None, fact))

    output_data = [asdict(e) for e in output_data]
    _write_items_to_jsonl(output_data, output_fname)


if __name__ == "__main__":
    output_path = "./data/evaluation/usr_convai2/"
    os.makedirs(output_path, exist_ok=True)
    download_usr_dataset(output_path)
    preprocess_usr_dataset(output_path + "pc_usr_data.json", output_path + "valid_processed.json")
    preprocess_usr_dataset(output_path + "pc_usr_data.json", output_path + "test_processed.json")
