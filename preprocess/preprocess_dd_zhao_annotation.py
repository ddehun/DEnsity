import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
from dataclasses import asdict, dataclass
from typing import List

import numpy as np

from evaluators.eval_utils import EvaluationExample

VALID_RATIO = 0.2


def _write_items_to_jsonl(data, fname):
    with open(fname, "w") as f:
        for el in data:
            f.write(json.dumps(el) + "\n")


def preprocess_zhao_dd(fname, output_fname):
    """Preprocessing the dataset from 'Designing Precise and Robust Dialogue Response Evaluators' with the same preprocessing setup."""
    with open(fname, "r") as f:
        data = json.load(f)
    output_data = []

    for item in data.values():
        context = [e[1].strip() for e in item["context"] if e[1].strip() != ""]
        answer = item["reference"][1].strip()
        responses = item["responses"]
        for k, v in responses.items():
            response = v["uttr"]
            scores = [int(e["overall"]) for e in list(v["scores"].values())]
            outliers = _detect_by_abd_median(scores, 1)
            scores = [s for s in scores if s not in outliers]
            assert all([1 <= e <= 5 for e in scores])
            if len(scores) == 0:
                continue
            score = float(np.mean([(score - 1) / 4 for score in scores]))
            output_data.append(EvaluationExample(context, answer, response, score, k, None))
    output_data = [asdict(e) for e in output_data]
    lens = len(output_data)
    valid_data = output_data[: int(lens * VALID_RATIO)]
    test_data = output_data[int(lens * VALID_RATIO) :]
    _write_items_to_jsonl(valid_data, output_fname.format("valid"))
    _write_items_to_jsonl(test_data, output_fname.format("test"))


def preprocess_usr_dataset(fname, output_fname):
    with open(fname, "r") as f:
        # data = [json.loads(s) for s in f.readlines()]
        data = json.load(f)
    output_data = []
    for el in data:
        context = [e.strip() for e in el["context"].split("\n") if e.strip() != ""]
        fact = [e.strip() for e in el["fact"].split("\n") if e.strip() != ""]
        reference = [
            single_response["response"].strip() for single_response in el["responses"] if single_response["model"] == "Original Ground Truth"
        ]
        assert len(reference) == 1
        reference = reference[0]
        for item in el["responses"]:
            response = item["response"].strip()
            model = item["model"]
            scores = item["Overall"]
            assert all([1 <= e <= 5 for e in scores])
            score = float(np.mean([(score - 1) / 4 for score in scores]))
            output_data.append(EvaluationExample(context, reference, response, score, model, fact))
    output_data = [asdict(e) for e in output_data]
    lens = len(output_data)
    valid_data = output_data[: int(lens * VALID_RATIO)]
    test_data = output_data[int(lens * VALID_RATIO) :]
    _write_items_to_jsonl(valid_data, output_fname.format("valid"))
    _write_items_to_jsonl(test_data, output_fname.format("test"))


def _detect_by_abd_median(data, n_dev):
    '''from "Detecting outliers: Do not use standard deviation around the mean, use absolute deviation around the median"'''
    if len(data) == 0:
        return []

    # Set upper and lower limit to n_dev absolute deviation medians
    data_median = np.median(data)
    abs_deviation = np.abs(data - data_median)
    abs_deviation_median = np.median(abs_deviation)
    b = 1.4826
    mad = abs_deviation_median * b
    anomaly_cut_off = mad * n_dev

    lower_limit = data_median - anomaly_cut_off
    upper_limit = data_median + anomaly_cut_off

    # Generate outliers
    outliers = []
    for data_point in data:
        if data_point > upper_limit or data_point < lower_limit:
            outliers.append(data_point)
    return outliers


if __name__ == "__main__":
    preprocess_zhao_dd("data/evaluation/dd/dd_annotations.json", "data/evaluation/dd/{}_processed.json")
