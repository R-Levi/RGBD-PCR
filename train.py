import logging
import os
import random
import time

import hydra
import numpy as np
import pytorch_lightning as zeus
import torch
from omegaconf import DictConfig, OmegaConf

from src.datasets import build_loader
from src.nnutils.rgbd_trainer import RGBD_Registration
from src.utils.util import makedir


@hydra.main(config_name="debug_config", config_path="src/configs")
def train(cfg: DictConfig) -> None:

    # Reproducibility - refer https://pytorch.org/docs/stable/notes/randomness.html
    # torch.autograd.set_detect_anomaly(True)
    torch.manual_seed(7351)
    # torch.set_default_dtype(torch.float32)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    random.seed(7351)
    np.random.seed(7351)

    assert cfg.experiment.name != "", "Experiment name is not defined."
    exp_time = time.strftime("%m%d%H%M")
    full_exp_name = f"{cfg.experiment.name}_{exp_time}"
    OmegaConf.set_struct(cfg, False)
    cfg.experiment.full_name = full_exp_name
    OmegaConf.set_struct(cfg, True)

    print("=====================================")
    print(f"Experiment name: {full_exp_name}")
    print()
    print(OmegaConf.to_yaml(cfg))
    print("=====================================")

    # setup checkpoint directory
    exp_dir = os.path.join(cfg.paths.experiments_dir, full_exp_name)
    makedir(exp_dir)

    train_loader,neighbor_limits = build_loader(cfg.dataset, split="train",neighbor_limits=None)
    valid_loader,neighbor_limits = build_loader(cfg.dataset, split="valid",neighbor_limits=neighbor_limits)

    # Trainer Plugins
    checkpoint_callback = zeus.callbacks.ModelCheckpoint(
        dirpath=exp_dir, filename="checkpoint-{epoch:03d}-{step:07d}", save_top_k=-1,
    )
    logger = zeus.loggers.TensorBoardLogger(
        save_dir=cfg.paths.tensorboard_dir, name=cfg.experiment.name, version=exp_time,
    )
    # Set up Trainer
    trainer = zeus.Trainer(
        gpus=cfg.system.num_gpus,
        num_sanity_val_steps=0,
        logger=logger,
        val_check_interval=cfg.train.eval_step,
        max_steps=cfg.train.num_steps,
        callbacks=[checkpoint_callback],
    )

    model = RGBD_Registration(cfg)

    # train model
    trainer.validate(model, valid_loader)
    trainer.fit(model, train_loader, valid_loader)


if __name__ == "__main__":
    train()
