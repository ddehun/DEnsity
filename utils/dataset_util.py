import os
import pickle
import random
from dataclasses import asdict, dataclass
from functools import partial
from typing import List, Tuple

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import AutoTokenizer

from utils.config import DATASET_TO_KEYS, TURN_TOKEN
from utils.utils import read_jsonl


@dataclass
class SelectionExample:
    history: List[str]
    answer: str
    negative_candidates: List[str]


@dataclass
class FeaturizedEvaluationExample:
    history: List[str]
    answer: str
    response: str
    score: float
    modelname: str = None
    fact: List[str] = None
    featurized_response: List[int] = None
    featurized_answer: List[int] = None
    response_repr: torch.Tensor = None
    answer_repr: torch.Tensor = None


def selection_collate_fn(samples, pad_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
    assert isinstance(pad_id, int)
    if (
        isinstance(samples[0], FeaturizedEvaluationExample) and "featurized_response" in asdict(samples[0]).keys()
    ):  # quality dataset. input_ids: bs* 2(ref and answer) *seq len (wo pad)
        input_ids_list = [torch.tensor(seq) for e in samples for seq in [e.featurized_response, e.featurized_answer]]
        mask_list = list(map(partial(torch.ones, dtype=torch.long), list(map(len, input_ids_list))))
    else:  # original dataset. input_ids: bs * (1+num_neg) * seq len (wo pad)
        input_ids_list = list(map(torch.tensor, sum(list(map(lambda x: x["input_ids"], samples)), [])))
        mask_list = list(map(torch.tensor, sum(list(map(lambda x: x["attention_mask"], samples)), [])))
    input_ids = pad_sequence(input_ids_list, batch_first=True, padding_value=pad_id)
    mask = pad_sequence(mask_list, batch_first=True, padding_value=0)
    return input_ids, mask


def load_dataset_for_reranker(
    dataset_name: str,
    setname: str,
    tokenizer,
    max_seq_len: int,
    tensor_name: str,
):
    assert setname in ["train", "valid", "test"]
    if dataset_name == "dd":
        datasetclass = DailyDialogforSelection
    elif dataset_name == "convai2":
        datasetclass = ConvAI2forSelection
    else:
        raise NotImplementedError(dataset_name)

    dataset = datasetclass(
        DATASET_TO_KEYS[dataset_name][f"{setname}_fname"],
        tokenizer,
        max_seq_len,
        DATASET_TO_KEYS[dataset_name]["num_negative"],
        tensor_name,
    )
    return dataset


class SelectionDataset(Dataset):
    def __init__(
        self,
        fname: str,
        tokenizer,
        max_seq_len: int,
        num_neg_cands: int,
        tensor_fname: str,
    ):
        self.do_init(
            fname,
            tokenizer,
            max_seq_len,
            num_neg_cands,
            tensor_fname,
        )

    def do_init(
        self,
        fname: str,
        tokenizer,
        max_seq_len: int,
        num_neg_cands: int,
        tensor_save_name: str,
    ):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.num_neg_cands = num_neg_cands

        print(f"Tensor name: {tensor_save_name}")
        raw_dataset = self._read_raw_files(fname)
        self.selection_examples = self._make_selection_examples(raw_dataset)

        if os.path.exists(tensor_save_name):
            print("Load Tensorized Feature...")
            features = self._load_features(tensor_save_name)
            print("End!")
        else:
            features = self._make_feature(self.selection_examples)
            self._save_features(features, tensor_save_name)
        self.features = features

    def _load_features(self, fname):
        with open(fname, "rb") as f:
            return pickle.load(f)

    def _save_features(self, feature, fname):
        with open(fname, "wb") as f:
            pickle.dump(feature, f)

    def __getitem__(self, idx):
        feature = self.features[idx]
        feature["index"] = idx
        return feature

    def __len__(self):
        return len(self.features)

    def _make_feature(self, examples):
        """
        Tensorize each elements
        """
        feature_list = []
        print("Tensorize...")
        context_length_list = []
        for idx, el in enumerate(tqdm(examples)):
            conv, ans, negs = el.history, el.answer, el.negative_candidates
            conv = TURN_TOKEN.join(conv)
            token_conv = self.tokenizer.tokenize(conv)

            max_res_len = max(map(len, list(map(self.tokenizer.tokenize, [ans] + negs))))
            context_length_list.append(len(token_conv) + max_res_len)
            # Truncate long histories from left-first, since the right-most utterance is more important for selection task.
            token_conv = token_conv[-(self.max_seq_len - 3 - max_res_len) :]
            truncated_conv = self.tokenizer.convert_tokens_to_string(token_conv)
            assert self.num_neg_cands == len(negs)
            encoded = self.tokenizer(
                [truncated_conv] * (1 + self.num_neg_cands),
                [ans] + negs,
                max_length=self.max_seq_len,
                truncation=True,
                padding=True,
            )
            if "token_type_ids" in encoded:
                del encoded["token_type_ids"]

            encoded["label"] = 0
            feature_list.append(encoded)
        print(f"Mean: {sum(context_length_list) / len(context_length_list)}")
        return feature_list

    def _read_raw_files(self, fname):
        raise NotImplementedError

    def _make_selection_examples(self, raw_dataset) -> List[SelectionExample]:
        raise NotImplementedError


class DailyDialogforSelection(SelectionDataset):
    def _read_raw_files(self, fname) -> List[List[str]]:
        with open(fname, "r") as f:
            ls = [l.strip() for l in f.readlines()]
        total = []
        for line in ls:
            uttrs = [e.strip() for e in line.strip().split("__eou__") if e.strip() != ""]
            total.append(uttrs)
        return total

    def _make_selection_examples(self, raw_dataset):
        example_list = []
        all_uttrs = list(set([u for us in raw_dataset for u in us]))
        print("Example Contruction...")
        for el in tqdm(raw_dataset):
            for last_conv_idx in range(len(el) - 1):
                conv = el[: last_conv_idx + 1]
                res = el[last_conv_idx + 1]
                # Oversampling to avoid (1) in-context negatives, and (2) the same negative with the answer response
                negs = [n for n in random.sample(all_uttrs, self.num_neg_cands * 10) if n not in conv + [res]][: self.num_neg_cands]
                example_list.append(SelectionExample(conv, res, negs))
        return example_list


class ConvAI2forSelection(SelectionDataset):
    def _format_to_bce_loss(self, features):
        new_features = []
        for f in features:
            new_features.append(
                {
                    "input_ids": f["input_ids"][0],
                    "attention_mask": f["attention_mask"][0],
                    "label": 0,
                }
            )
            new_features.append(
                {
                    "input_ids": f["input_ids"][1],
                    "attention_mask": f["attention_mask"][1],
                    "label": 1,
                }
            )
        assert len(features) * 2 == len(new_features)
        return new_features

    def _read_raw_files(self, fname) -> List[List[str]]:
        with open(fname, "r") as f:
            ls = [l.strip() for l in f.readlines()]
        total = []
        tmp_conv = {"persona": [], "pairs": []}

        for idx, line in enumerate(ls):
            if line.startswith("1 your persona:") and idx != 0:
                total.append(tmp_conv)
                tmp_conv = {"persona": [], "pairs": []}
            line = line.split("\t")
            line[0] = " ".join(line[0].split(" ")[1:])
            assert len(line) in [1, 4]
            if len(line) == 1:
                tmp_conv["persona"].append(line[0])
            else:
                parter_uttr, my_uttr, _, candidates = line
                candidates = candidates.split("|")
                candidates.remove(my_uttr)
                tmp_conv["pairs"].append((parter_uttr, my_uttr, candidates))
        total.append(tmp_conv)
        return total

    def _make_selection_examples(self, raw_dataset):
        example_list = []
        print("Example Contruction...")
        for el in tqdm(raw_dataset):
            history = el["persona"]
            for turn in el["pairs"]:
                partner_uttr, my_uttr, candidates = turn
                history.append(partner_uttr)
                example_list.append(SelectionExample(history[:], my_uttr, candidates))
                history.append(my_uttr)
        return example_list


class QualityDataset(Dataset):
    def __init__(
        self,
        json_fname: str,
        tokenizer: AutoTokenizer = None,
        skip_human_response: bool = True,
        max_seq_len: int = 256,
    ):
        self.max_seq_len = max_seq_len
        example_list = read_jsonl(json_fname)
        if skip_human_response:
            example_list = self._skip_human_response(example_list)
        self.tokenizer = tokenizer
        self.examples, self.encoded = self._featurize(example_list)
        assert len(self.examples) == len(self.encoded)
        self.humanscore = list(map(lambda x: x.score, self.examples))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]

    def _skip_human_response(self, example_list):
        filtered_examples = []
        for el in example_list:
            if el["modelname"] not in ["ground-truth", "Original Ground Truth"]:
                filtered_examples.append(el)
        return filtered_examples

    def _featurize(self, examples):
        featurized_examples = []
        features = []
        for el in examples:
            ctx, ans, hyp = el["history"], [el["answer"]], [el["response"]]
            fact = el["fact"]
            if fact != [] and fact is not None:
                assert isinstance(fact, list) and len(fact) >= 1
                ctx = fact + ctx
            encoded = self.tokenizer(
                [TURN_TOKEN.join(ctx)] * 2,
                hyp + ans,
                max_length=self.max_seq_len,
                truncation=True,
                padding=False,
            )
            el["featurized_response"] = encoded["input_ids"][0]
            el["featurized_answer"] = encoded["input_ids"][1]
            features.append(encoded)
            featurized_examples.append(
                FeaturizedEvaluationExample(
                    **el,
                )
            )
        return featurized_examples, features
