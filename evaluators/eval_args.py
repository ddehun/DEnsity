import argparse
import os

from utils.config import LM_HIDDEN_DIM_MAP


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--random_seed", type=int, default=42)
    # Path arguments
    parser.add_argument("--save_path", type=str, default="logs")
    parser.add_argument(
        "--annotation_fname",
        type=str,
        default="./data/evaluation/{}/{}_processed.json",  # data, split(valid or test)
    )
    parser.add_argument(
        "--pickle_save_path",
        type=str,
        default="./results/pickle_save_path/{}/{}.pck",  # data, info of a pickled instance
    )
    parser.add_argument(
        "--repr_save_fname",
        type=str,
        default="./results/reprs/{}/{}.{}.npy",  # data, original or quality with split(train or valid or test), reranker_exp_name,
    )
    parser.add_argument(
        "--score_save_fname",
        type=str,
        default="./results/correlation/{}/{}.{}.{}.json",  # eval_data, train_data, expname, split
    )

    # Distance metrics
    parser.add_argument(
        "--distance_function",
        type=str,
        default="maha",
        choices=["maha"],
    )

    parser.add_argument("--ref_type", type=str, default="train", choices=["train"])

    # General arguments
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["dd", "convai2"],
    )
    # General arguments
    parser.add_argument(
        "--eval_dataset",
        type=str,
        choices=["dd", "convai2", "fed", "grade_eval_dd", "usr_convai2", "grade_eval_convai2"],
        default="dd",
    )
    parser.add_argument("--eval_batch_size", type=int, default=64)
    parser.add_argument("--max_seq_len", type=int, default=256)
    parser.add_argument(
        "--lmtype",
        type=str,
        default="bert-base-uncased",
    )
    parser.add_argument(
        "--reranker_exp_name",
        type=str,
        default="reranker.vanilla.epoch10.lr5e-5",
    )
    parser.add_argument(
        "--tensor_fname",
        type=str,
        default="./data/tensors/selection.{}.{}.{}.maxlen{}.numneg{}.{}.pck",
        help="Path of the tensorized original dialogue dataset",
    )  # dataset, split, tokenizer, max_length, num_neg, random
    parser.add_argument(
        "--reranker_ckpt_path",
        type=str,
        default="./logs/{}/{}/models/bestmodel.pth",
    )  # args.dataset, args.reranker_exp_names
    parser.add_argument("--negative_type", type=str, default="random", choices=["random"])

    args = parser.parse_args()

    args.reranker_ckpt_path = args.reranker_ckpt_path.format(args.dataset, args.reranker_exp_name)
    args.hidden_repr_dim = LM_HIDDEN_DIM_MAP[args.lmtype]
    args.exp_setup = f"{args.distance_function}.ref-{args.ref_type}.reranker-{args.reranker_exp_name}"

    return args
