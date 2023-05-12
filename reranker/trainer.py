import os
from functools import partial

import numpy as np
import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast
from tqdm import tqdm

from utils.utils import get_mrr, get_recall1_for_reranker


class Trainer:
    def __init__(
        self,
        model,
        args,
        train_loader=None,
        eval_loader=None,
        optimizer=None,
        writer=None,
        scaler=None,
        lr_scheduler=None,
    ):
        self.model = model
        self.args = args
        self.train_loader = train_loader
        self.eval_loader = eval_loader
        self.optimizer = optimizer

        # Loss function in experiments
        self.ce_criteria = torch.nn.CrossEntropyLoss()
        self.scl_criteria = partial(self._scl_loss, temp=args.scl_temp)

        self.writer = writer
        self.scaler = scaler
        self.lr_scheduler = lr_scheduler
        self.device = torch.device("cuda")

        self.global_step = 0
        self.best_metric_score = 0

    def _scl_loss(self, reprs, labels, temp: float):
        norm_reprs = F.normalize(reprs, dim=-1)
        cosine_score = torch.exp(norm_reprs @ norm_reprs.t() / temp)
        cosine_score = cosine_score - torch.diag(
            torch.diag(cosine_score)
        )  
        mask = (labels.unsqueeze(1) == labels.unsqueeze(0)).float()
        mask = mask - torch.diag(torch.diag(mask)) 
        cos_loss = cosine_score / cosine_score.sum(
            dim=-1, keepdim=True
        )  
        cos_loss = -torch.log(cos_loss + 1e-5)
        cos_loss = (mask * cos_loss).sum(-1) / (mask.sum(-1) + 1e-3)
        cos_loss = cos_loss.mean()
        return cos_loss

    def train(self):
        self._save_ckpt(self.model, "begin_model.pth")
        self.label = torch.zeros(self.args.batch_size, dtype=torch.long).to(self.device)

        print(f"Train example: {len(self.train_loader)*self.args.batch_size}")
        print(f"Accumulation size: {self.args.accumulation_step}")
        print(f"Batch size: {self.args.batch_size}")
        print(f"Dataloader length: {len(self.train_loader)}")

        for epoch in range(self.args.epochs):
            self._train_epoch(epoch)

    def _get_scl_label(
        self, bs: int, num_negative: int, use_shared_negative_in_scl: bool = False
    ):
        """
        use_shared_negative_in_scl (bool): if True, assign the same label(1) to all negative pairs. This makes
        negative pairs within a batch to be pulled. Default: False
        """
        if use_shared_negative_in_scl:
            scl_label = (
                torch.cat(
                    [
                        torch.zeros([bs, 1]),
                        torch.ones([bs, num_negative]),
                    ],
                    1,
                )
                .reshape(-1)
                .to("cuda")
            )
        else:
            scl_label = (
                torch.cat(
                    [
                        torch.zeros([bs, 1]),
                        torch.arange(1, bs * num_negative + 1).reshape(bs, -1),
                    ],
                    1,
                )
                .reshape(-1)
                .to("cuda")
            )
        return scl_label

    def do_eval(self, epoch: int = None, ckpt_suffix: str = ""):
        self.model.eval()
        scl_loss_list = []
        ce_loss_list = []
        total_loss_list = []
        mrr_list = []
        recall_list = []
        for step, batch in enumerate(tqdm(self.eval_loader)):
            inp = (e.to(self.device, non_blocking=True) for e in batch)
            ids, mask = inp
            bs = int(len(ids) / (1 + self.args.num_negative))
            with torch.no_grad():
                output = self.model(ids, mask, return_hidden=self.args.use_scl_loss)
                if isinstance(output, tuple):
                    output, reprs = output
                    scl_label = self._get_scl_label(
                        bs, self.args.num_negative, self.args.use_shared_negative_in_scl
                    )

                output = output.squeeze().reshape(bs, 1 + self.args.num_negative)
                ce_loss = self.ce_criteria(output, self.label).cpu().numpy()
                ce_loss_list.append(ce_loss)
                total_loss = ce_loss

                if self.args.use_scl_loss:
                    scl_loss = self.scl_criteria(reprs, scl_label).cpu().numpy()
                    scl_loss_list.append(scl_loss)
                    total_loss += self.args.scl_coeff * scl_loss
            total_loss_list.append(total_loss)
            mrr = get_mrr(output.cpu().numpy(), do_average=False)
            mrr_list.extend(mrr)
            recall = get_recall1_for_reranker(output.cpu().numpy(), do_average=False)
            recall_list.extend(recall)

        valid_loss = np.mean(total_loss_list)
        valid_ce = np.mean(ce_loss_list)
        valid_recall = np.mean(recall_list)
        valid_mrr = np.mean(mrr_list)

        self.writer.add_scalar("loss/valid", valid_loss, self.global_step)
        self.writer.add_scalar("CE/valid", valid_ce, self.global_step)
        self.writer.add_scalar("MRR/valid", valid_mrr, self.global_step)
        self.writer.add_scalar("R1/valid", valid_recall, self.global_step)
        if self.args.use_scl_loss:
            valid_scl = np.mean(scl_loss_list)
            self.writer.add_scalar("SCL/train", valid_scl, self.global_step)

        if epoch is not None:
            self._save_ckpt(self.model, f"epoch{epoch}{ckpt_suffix}.pth")
            if valid_recall > self.best_metric_score:
                self.best_metric_score = valid_recall
                self._save_ckpt(self.model, "bestmodel.pth")

    def _train_epoch(self, current_epoch: int):
        progress_bar = tqdm(range(self.args.num_update_steps_per_epoch))
        self.model.train()
        labels = torch.zeros(self.args.batch_size, dtype=torch.long).to(self.device)
        # global_step = self.args.num_update_steps_per_epoch * current_epoch
        for step, batch in enumerate(self.train_loader):
            inp = (e.to(self.device, non_blocking=True) for e in batch)
            ids, mask = inp
            bs = int(len(ids) / (1 + self.args.num_negative))
            assert bs == self.args.batch_size
            with autocast():
                output = self.model(ids, mask, return_hidden=self.args.use_scl_loss)
                if isinstance(output, tuple):
                    output, reprs = output
                    scl_label = self._get_scl_label(
                        bs, self.args.num_negative, self.args.use_shared_negative_in_scl
                    )
                output = output.squeeze().reshape(bs, 1 + self.args.num_negative)
                if self.args.skip_ce_loss:
                    ce_loss, loss = 0, 0
                else:
                    ce_loss = self.ce_criteria(output, labels)
                    loss = ce_loss

                if self.args.use_scl_loss:
                    scl_loss = self.scl_criteria(reprs, scl_label)
                    loss = loss + self.args.scl_coeff * scl_loss
                loss = loss / self.args.accumulation_step

            self.scaler.scale(loss).backward()
            # https://pytorch.org/docs/stable/notes/amp_examples.html#gradient-accumulation
            if (
                step % self.args.accumulation_step == 0
                or step == len(self.train_loader) - 1
            ):
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.args.grad_norm
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.lr_scheduler.step()
                self.optimizer.zero_grad()

                progress_bar.update(1)
                self.global_step += 1

            if step % 100 == 0:
                # Tensorboard
                mrr = get_mrr(output)
                recall = get_recall1_for_reranker(output)
                self.writer.add_scalar(
                    "loss/train", loss * self.args.accumulation_step, self.global_step
                )
                self.writer.add_scalar("CE/train", ce_loss, self.global_step)
                self.writer.add_scalar("MRR/train", mrr, self.global_step)
                self.writer.add_scalar("R1/train", recall, self.global_step)
                if self.args.use_scl_loss:
                    self.writer.add_scalar("SCL/train", scl_loss, self.global_step)

            if step in [int(0.5 * len(self.train_loader)), len(self.train_loader) - 1]:
                suffix = "_half" if step == int(0.5 * len(self.train_loader)) else ""
                self.do_eval(current_epoch, suffix)
        progress_bar.close()

    def _save_ckpt(self, model, fname):
        output_path = os.path.join(self.args.exp_path, "models", fname)
        torch.save(
            model.module.state_dict(),
            output_path,
        )
