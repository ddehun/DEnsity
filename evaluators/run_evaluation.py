import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import transformers

from evaluators.eval_args import get_args
from evaluators.eval_utils import save_prediction_output, get_conv_repr, get_correlation
from evaluators.model import DEnsity
from utils.config import DATASET_TO_KEYS
from utils.dataset_util import QualityDataset, load_dataset_for_reranker
from utils.utils import load_tokenizer_and_reranker, read_npy, write_npy


def main(args):
    transformers.logging.set_verbosity_error()
    tokenizer, reranker = load_tokenizer_and_reranker(args.lmtype, args.reranker_ckpt_path)
    mahalanobis_pck_name = args.pickle_save_path.format(args.dataset, f"{args.exp_setup}.positive")

    print(f" [*] Experiment name: {args.exp_setup}")
    print(f" [*] Evaluation Dataset: {args.eval_dataset}")
    print(f" [*] Humen feature fname: {mahalanobis_pck_name}")

    """
    Load datasets
    """
    datasets = {
        "quality.test": QualityDataset(
            args.annotation_fname.format(args.eval_dataset, "test"),
            tokenizer,
            max_seq_len=args.max_seq_len,
        ),
    }

    if not os.path.exists(mahalanobis_pck_name):
        pickled_dataset_fname = args.tensor_fname.format(
            args.dataset,
            "train",
            args.lmtype,
            args.max_seq_len,
            DATASET_TO_KEYS[args.dataset]["num_negative"],
            args.negative_type,
        )
        datasets["original.train"] = load_dataset_for_reranker(
            dataset_name=args.dataset, setname="train", tokenizer=tokenizer, max_seq_len=args.max_seq_len, tensor_name=pickled_dataset_fname
        )

    """
    Feature Extraction (Both human and chatbot responses)
    """

    for name, dataset in datasets.items():
        repr_data = args.eval_dataset if "quality." in name else args.dataset
        repr_save_fname = args.repr_save_fname.format(repr_data, name, args.reranker_exp_name)
        settype = name.split(".")[0]
        reprs = get_conv_repr(
            args,
            dataset,
            settype,
            tokenizer,
            reranker,
        )
        write_npy(reprs, repr_save_fname)

        # if os.path.exists(repr_save_fname):
        #     reprs = read_npy(repr_save_fname)
        # else:
        #     settype = name.split(".")[0]
        #     reprs = get_conv_repr(
        #         args,
        #         dataset,
        #         settype,
        #         tokenizer,
        #         reranker,
        #     )
        #     write_npy(reprs, repr_save_fname)

        assert len(reprs) == len(dataset)
        datasets[name].reprs = reprs

    """
    Evaluation
    """
    hyp_repr = torch.tensor(datasets["quality.test"].reprs)
    human_repr = torch.tensor(datasets["original.train"].reprs[:, :1]) if not os.path.exists(mahalanobis_pck_name) else None

    evaluator = DEnsity(
        human_repr,
        mean_cov_pck_fname=mahalanobis_pck_name,
    )
    pred = evaluator.infer(hyp_repr)
    human_score = [example.score for example in datasets["quality.test"].examples]
    correlation = get_correlation(human_score, pred)
    assert len(pred) == len(human_score)
    print(" [**] Evaluation Result")
    print(correlation)

    output_fname = args.score_save_fname.format(args.eval_dataset, args.dataset, args.exp_setup + ".pos", "quality.test")
    save_prediction_output(output_fname, datasets["quality.test"].examples, pred)


if __name__ == "__main__":
    args = get_args()
    main(args)
