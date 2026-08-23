import logging
import os

import torch
from lerna.utils import instantiate
from lerna.core.hydra_config import HydraConfig
from torch.utils.tensorboard import SummaryWriter

from optim_env.torch_env.utils.metrics import accuracy
from optim_env.torch_env.utils.checkpointing import save_checkpoint


class SGDTrainer:
    # dataset.framework must be "torch" (see the shared DataModule) so
    # get_train_loader/get_val_loader yield torch tensors.
    def __init__(self, cfg):
        self.cfg = cfg
        self.logger = logging.getLogger(cfg.exp_name)
        output_log_dir = HydraConfig.get().runtime.output_dir
        self.writer = SummaryWriter(output_log_dir)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.manual_seed(cfg.seed)
        self.logger.info(f"Using device: {self.device}")

        self.logger.info(" => Creating dataset ...")
        self.prepare_dataset()
        self.logger.info(" => Creating model ...")
        self.prepare_model()
        self.logger.info(" => Creating loss fn ...")
        self.prepare_loss_fn()
        self.best_loss = float("inf")

    def fit(self):
        for self.epoch in range(self.cfg.epochs):
            epoch_train_loss = self.run_train_epoch()

            self.logger.info(f"Epoch {self.epoch} | train={epoch_train_loss:.4f}")
            self.writer.add_scalar("train/loss_epoch", epoch_train_loss, self.epoch)

            # Evaluation
            val_loss, val_acc = self.evaluate()
            self.writer.add_scalar("val/loss", val_loss, self.epoch)
            if val_acc is not None:
                self.writer.add_scalar("val/accuracy", val_acc, self.epoch)

            self.logger.info(
                f"Epoch {self.epoch} | val loss={val_loss:.4f}"
                + (f" accuracy={val_acc:.4f}" if val_acc is not None else "")
            )

            # Track best model in memory
            if val_loss < self.best_loss:
                self.best_loss = val_loss
                self.best_state_dict = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
                self.logger.info(f"Epoch {self.epoch} | New best val loss: {val_loss:.4f}")

        if hasattr(self, "best_state_dict") and self.best_state_dict is not None:
            checkpoint_path = os.path.join(self.cfg.save_path, "best_model.pt")
            save_checkpoint(self.best_state_dict, checkpoint_path)
            self.logger.info(f"Final best model saved to {checkpoint_path}")

        self.writer.close()

    def run_train_epoch(self):
        self.model.train()
        losses = []
        for i, (x, y) in enumerate(self.train_loader):
            x, y = x.to(self.device), y.to(self.device)

            self.optimizer.zero_grad()
            pred = self.model(x)
            loss = self.loss_fn(pred, y)
            loss.backward()
            self.optimizer.step()

            loss_value = loss.item()
            losses.append(loss_value)
            self.writer.add_scalar("train/loss_step", loss_value, self.epoch * len(self.train_loader) + i)
        return sum(losses) / len(losses)

    def evaluate(self):
        self.model.eval()
        losses = []
        accs = []
        is_classification = self.cfg.dataset.num_classes and self.cfg.dataset.num_classes > 0
        with torch.no_grad():
            for x, y in self.val_loader:
                x, y = x.to(self.device), y.to(self.device)
                pred = self.model(x)
                loss = self.loss_fn(pred, y)
                losses.append(loss.item())

                if is_classification and (y.ndim == 1 or (y.ndim == 2 and y.shape[1] > 1)):
                    accs.append(accuracy(pred, y).item())

        avg_loss = sum(losses) / len(losses)
        avg_acc = sum(accs) / len(accs) if accs else None
        return avg_loss, avg_acc

    def prepare_dataset(self):
        dataset_manager = instantiate(self.cfg.dataset)
        self.train_ds = dataset_manager.get_train_ds()
        self.val_ds = dataset_manager.get_val_ds()
        self.train_loader = dataset_manager.get_train_loader(self.train_ds)
        self.val_loader = dataset_manager.get_val_loader(self.val_ds)

    def prepare_model(self):
        self.model = instantiate(self.cfg.model).to(self.device)
        # torch.optim optimizers need model.parameters() at construction time, passed in
        # as an extra kwarg on top of whatever the config itself specifies (e.g. lr).
        self.optimizer = instantiate(self.cfg.optimizer, params=self.model.parameters())

    def prepare_loss_fn(self):
        self.loss_fn = instantiate(self.cfg.loss)
