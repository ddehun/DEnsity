"""
Train script to learn a cross-encoder
"""
import argparse
import math
import os
import sys
from functools import partial

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch import nn
from torch.cuda.amp import GradScaler
from torch.nn import functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoTokenizer, get_scheduler, logging

from reranker.models import DefaultSelectionModel
from reranker.trainer import Trainer
from utils.config import DATASET_TO_KEYS
from utils.dataset_util import (TURN_TOKEN, load_dataset_for_reranker,
                                selection_collate_fn)
from utils.utils import dump_config, set_seed, setup_path


def main(args):
    logging.set_verbosity_warning()
    set_seed(args.random_seed)
    setup_path(args.exp_path)
    dump_config(args)

    tokenizer = AutoTokenizer.from_pretrained(args.lmtype)
    tokenizer.add_special_tokens({"additional_special_tokens": [TURN_TOKEN]})

    """
    Dataset load
    """
    args.num_negative = DATASET_TO_KEYS[args.dataset]["num_negative"]
    train_dataset = load_dataset_for_reranker(
        args.dataset,
        "train",
        tokenizer,
        args.max_seq_len,
        args.tensor_fname.format(
            args.dataset,
            "train",
            args.lmtype,
            args.max_seq_len,
            args.num_negative,
            args.negative_type,
        ),
    )
    valid_dataset = load_dataset_for_reranker(
        args.dataset,
        "valid",
        tokenizer,
        args.max_seq_len,
        args.tensor_fname.format(
            args.dataset,
            "valid",
            args.lmtype,
            args.max_seq_len,
            args.num_negative,
            args.negative_type,
        ),
    )

    train_partial_selection_collate_fn = partial(selection_collate_fn, pad_id=tokenizer.pad_token_id)

    valid_partial_selection_collate_fn = partial(selection_collate_fn, pad_id=tokenizer.pad_token_id)

    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=args.batch_size,
        collate_fn=train_partial_selection_collate_fn,
        drop_last=True,
    )
    valid_loader = DataLoader(
        valid_dataset,
        shuffle=False,
        batch_size=args.batch_size,
        collate_fn=valid_partial_selection_collate_fn,
        drop_last=True,
    )

    """
    Tarining setup
    """
    device = torch.device("cuda")
    model = DefaultSelectionModel(args.lmtype)
    model.model.resize_token_embeddings(len(tokenizer))
    model = torch.nn.DataParallel(model)
    model = model.to(device)
    scaler = GradScaler()
    assert not args.wo_fp16

    # Optimizer
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.lr)

    # Training steps
    num_update_steps_per_epoch = math.ceil(len(train_loader) / args.accumulation_step)
    args.num_update_steps_per_epoch = num_update_steps_per_epoch
    if args.scheduler_type == "linear":
        args.max_train_steps = args.epochs * num_update_steps_per_epoch
        lr_scheduler = get_scheduler(
            name=args.scheduler_type,
            optimizer=optimizer,
            num_warmup_steps=args.num_warmup_steps,
            num_training_steps=args.max_train_steps,
        )
    elif args.scheduler_type == "plateau":
        lr_scheduler = ReduceLROnPlateau(
            optimizer,
            "max",
            factor=args.plateau_factor,
            patience=args.plateau_patience,
            verbose=True,
        )
    else:
        raise ValueError(args.scheduler_type)

    writer = SummaryWriter(os.path.join(args.exp_path, "board"))

    """
    Training
    """
    trainer = Trainer(
        model,
        args,
        train_loader,
        valid_loader,
        optimizer,
        writer,
        scaler,
        lr_scheduler,
    )
    trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Path and Dataset
    parser.add_argument("--dataset", type=str, choices=["dd", "convai2", "personachat"], default="dd")
    parser.add_argument("--save_path", type=str, default="logs")
    parser.add_argument("--exp_name", type=str, default="vanilla")
    parser.add_argument(
        "--tensor_fname",
        type=str,
        default="./data/tensors/selection.{}.{}.{}.maxlen{}.numneg{}.{}.pck",
    )  # dataset, split, tokenizer, max_length, num_neg, random

    # Basic hyperparameters
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument("--max_seq_len", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--accumulation_step", type=int, default=2)
    parser.add_argument("--wo_fp16", action="store_true")
    parser.add_argument("--grad_norm", type=float, default=1.0)
    parser.add_argument("--lmtype", type=str, default="bert-base-uncased")
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--scheduler_type", type=str, default="linear", choices=["plateau", "linear"])
    parser.add_argument("--negative_type", type=str, default="random", choices=["random"])
    parser.add_argument("--num_warmup_steps", type=int, default=1000)

    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay to use.")
    parser.add_argument("--plateau_factor", type=float, default=0.4)
    parser.add_argument("--plateau_patience", type=float, default=0)

    # SCL parameters
    parser.add_argument("--skip_ce_loss", action="store_true")
    parser.add_argument("--use_scl_loss", action="store_true")
    parser.add_argument("--use_shared_negative_in_scl", action="store_true")
    parser.add_argument("--scl_coeff", type=float, default=1.0)
    parser.add_argument("--scl_temp", type=float, default=1.0)
    args = parser.parse_args()

    args.exp_path = os.path.join(args.save_path, args.dataset, args.exp_name)
    print(args.exp_path)
    main(args)
