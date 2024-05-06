import os
import random
import hydra
import torch
import numpy as np
import pytorch_lightning as zeus
from omegaconf import DictConfig, OmegaConf
from src.datasets import build_loader
from src.nnutils.rgbd_trainer import RGBD_Registration


@hydra.main(config_name="debug_config", config_path="src/configs")
def test(cfg: DictConfig) -> None:

    torch.manual_seed(7351)
    random.seed(7351)
    np.random.seed(7351)

    # --- load checkpoint ---
    ckpt_cfg = cfg.test.checkpoint
    if ckpt_cfg.name == "":
        # assume no checkpoint and run with untrained model
        exp_name = cfg.experiment.name
        exp_time = ckpt_cfg.time
        OmegaConf.set_struct(cfg, False)
        cfg.experiment.full_name = f"{exp_name}_{exp_time}"
        model = RGBD_Registration(cfg)
    else:
        exp_name = ckpt_cfg.name
        exp_time = ckpt_cfg.time
        ckpt_dir = os.path.join(
            cfg.paths.experiments_dir, f"{ckpt_cfg.name}_{ckpt_cfg.time}"
        )
        print(ckpt_dir)

        if ckpt_cfg.step == -1:
            ckpts = os.listdir(ckpt_dir)
            ckpts.sort()
            # pick last file by default -- most recent checkpoint
            ckpt_file = ckpts[-1]
            print(f"Using the last checkpoint: {ckpt_file}")
        else:
            epoch = ckpt_cfg.epoch
            step = ckpt_cfg.step
            ckpt_file = f"checkpoint-epoch={epoch:03d}-step={step:07d}.ckpt"
            print(f"Using the last checkpoint: {ckpt_file}")

        checkpoint_path = os.path.join(ckpt_dir, ckpt_file)

        # model = BYOC_Registration.load_from_checkpoint(checkpoint_path, strict=False)
        model = RGBD_Registration(cfg)
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint["state_dict"])

    # get dataset split and get first item, useful when debugging
    loader,neighbor_limits = build_loader(cfg.dataset, split='test',neighbor_limits=[89, 31, 33, 34])
    print("neighbor_limits:",neighbor_limits)
    loader.dataset.__getitem__(0)

    # -- test model --
    trainer = zeus.Trainer(gpus=1)
    trainer.test(model, loader, verbose=False)

if __name__ == "__main__":
    test()















