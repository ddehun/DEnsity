import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


import os
import pickle
import sys
from typing import List

import numpy as np
import torch
from sklearn.covariance import EmpiricalCovariance


class DEnsity:
    def __init__(self, reprs, mean_cov_pck_fname: str, tokenizer=None, reranker=None, turn_token: str = "[SEPT]", max_length: int = 256):
        self.mean_cov_pck_fname = mean_cov_pck_fname
        self.mean, self.var = self._get_mean_and_var(reprs)
        self.tokenizer, self.reranker = tokenizer, reranker
        self.turn_token = turn_token
        self.max_length = max_length

    def _get_mean_and_var(self, reprs):
        if os.path.exists(self.mean_cov_pck_fname):
            with open(self.mean_cov_pck_fname, "rb") as f:
                return pickle.load(f)
        else:
            mean = reprs.mean(0)
            var = EmpiricalCovariance().fit(reprs - mean).precision_.astype(np.float32)
            os.makedirs(os.path.dirname(self.mean_cov_pck_fname), exist_ok=True)
            with open(self.mean_cov_pck_fname, "wb") as f:
                pickle.dump([mean, var], f)
        return mean, var

    def _get_mahalanobis_distance(self, mean, var, hyp_repr):
        centered_hyp_repr = hyp_repr - np.expand_dims(mean, axis=0)
        dist = np.diagonal(np.matmul(np.matmul(centered_hyp_repr, var), centered_hyp_repr.T))
        assert len(dist) == len(hyp_repr)
        return dist

    def _get_euclidean_distance(self, mean, hyp_repr):
        centered_hyp_repr = hyp_repr - np.expand_dims(mean, axis=0)
        dist = np.diagonal(np.matmul(centered_hyp_repr, centered_hyp_repr.T))
        assert len(dist) == len(hyp_repr)
        return dist

    def infer(self, hyp_reprs, is_euclidean: bool = False) -> List[float]:
        if is_euclidean:
            dist = self._get_euclidean_distance(self.mean, hyp_reprs)
        else:
            dist = self._get_mahalanobis_distance(self.mean, self.var, hyp_reprs)
        return -np.sqrt(dist)

    @torch.no_grad()
    def _get_single_dialogue_repr(self, dialogue: List[str]) -> np.ndarray:
        encoded = self.tokenizer(dialogue[0], dialogue[1], return_tensors="pt", padding=False, truncation="only_first")
        encoded = {k: v.to("cuda") for k, v in encoded.items()}

        _, repr = self.reranker(encoded["input_ids"], encoded["attention_mask"], return_hidden=True)
        return repr.cpu().numpy()

    def evaluate(self, dialogue: List[str], is_turn_level: bool = True):
        assert isinstance(dialogue, list) and all([isinstance(uttr, str) for uttr in dialogue]) and len(dialogue) >= 2
        assert self.tokenizer is not None and self.reranker is not None

        if is_turn_level:  # Evaluate the last turn only
            context, response = self.turn_token.join(dialogue[:-1]), dialogue[-1]
            eval_seq = [[context, response]]
        else:  # Evaluate individual turns in the dialogue and averaging
            eval_seq = []
            for turn_idx, turn in enumerate(dialogue):
                if turn_idx == 0:
                    continue
                eval_seq.append([self.turn_token.join(dialogue[:turn_idx]), turn])

        distances = []
        for pair in eval_seq:
            pair_repr = self._get_single_dialogue_repr(pair)
            distance = self._get_mahalanobis_distance(self.mean, self.var, pair_repr)[0]
            distances.append(distance)

        return -np.sqrt(np.mean(distances))
