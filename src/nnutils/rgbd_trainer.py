import pytorch_lightning as zeus
import torch
from ..models import build_model
from ..utils.util import detach_dictionary
from .loss import OverallLoss,Evaluator


class RGBD_Registration(zeus.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        # Build model
        self.model = build_model(cfg.model)
        self.num_views = cfg.dataset.num_views

        #geo_loss
        self.loss_func = OverallLoss(cfg).cuda()
        self.evaluator = Evaluator(cfg).cuda()

    def calculate_loss_and_metrics(self, batch, output, train=False):
        # evaluate losses and metrics
        loss, metrics = [], {}

        if not train:
            metrics = self.evaluator(output,batch)
            return metrics
        else:
            metrics = self.evaluator(output, batch)
            geo_loss = self.loss_func(output, batch)
            print(f"loss:{geo_loss['loss']},c_loss:{geo_loss['c_loss']},f_closs:{geo_loss['f_loss']}")
            print("lr:",self.optimizers().optimizer.state_dict()['param_groups'][0]['lr'])
            return geo_loss['loss'], metrics



    def training_step(self, batch, batch_idx):
        return self.forward_step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx):
        return self.forward_step(batch, batch_idx, "valid")

    def test_step(self, batch, batch_idx):
        return self.forward_step(batch, batch_idx, "test")

    def forward_step(self, batch, batch_idx, split):
        # forward pass
        output = self.model(batch)
        is_train = split == "train"
        if is_train:
            loss, metrics = self.calculate_loss_and_metrics(batch, output, train=is_train)
            self.log(f"loss/{split}", loss)
            output = detach_dictionary(output)
            return {
                "loss": loss,
                "output": output,
            }
        else:
            metrics = self.calculate_loss_and_metrics(batch, output, train=is_train)
            for f_type in metrics:
                val = metrics[f_type]
                self.log(f"{f_type}/{split}", val)
            metrics = detach_dictionary(metrics)

            return {
                "metrics": metrics,
            }

    def configure_optimizers(self):
        params = self.model.parameters()
        cfg = self.cfg.train
        optimizer = torch.optim.Adam(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer,cfg.lr_decay_steps, gamma=cfg.lr_decay)
        return [optimizer],[{"scheduler":scheduler,"interval":"step"}]



    def test_epoch_end(self, test_step_outputs):
        error_R = []
        error_t = []

        for dict in test_step_outputs:
            error_R.append(dict["metrics"]['RRE'])
            error_t.append(dict["metrics"]['RTE'])

        error_R = torch.tensor(error_R)
        error_t = torch.tensor(error_t) * 100

        recall_R = [(error_R <= thresh).float().mean() for thresh in [5, 10, 45]]
        recall_t = [(error_t <= thresh).float().mean() for thresh in [5, 10, 25]]

        # save result
        # log_name = self.cfg.experiment.name + '.txt'
        out = f"Pairwise Registration:\n " + \
              f"rotation\n" \
              f"5\t  10\t  45\n" + \
              f"{recall_R[0] * 100:4.1f}\t" + \
              f"{recall_R[1] * 100:4.1f}\t" + \
              f"{recall_R[2] * 100:4.1f}\t" + \
              f"{error_R.mean():4.1f}\t" + \
              f"{error_R.median():4.1f} \n" + \
              f"translation:\n" + \
              f"5\t  10\t  25\n" + \
              f"{recall_t[0] * 100:4.1f}\t" + \
              f"{recall_t[1] * 100:4.1f}\t" + \
              f"{recall_t[2] * 100:4.1f}\t" + \
              f"{error_t.mean():4.1f} \t" + \
              f"{error_t.median():4.1f}"
        # with open(log_name, 'w') as f:
        #     f.write(out)
        # f.close()
        # print(log_name)

        print(
            "Pairwise Registration:   ",
            f"{recall_R[0] * 100:4.1f} ",
            f"{recall_R[1] * 100:4.1f} ",
            f"{recall_R[2] * 100:4.1f} ",
            f"{error_R.mean():4.1f} ",
            f"{error_R.median():4.1f} ",
            " || ",
            f"{recall_t[0] * 100:4.1f} ",
            f"{recall_t[1] * 100:4.1f} ",
            f"{recall_t[2] * 100:4.1f} ",
            f"{error_t.mean():4.1f} ",
            f"{error_t.median():4.1f} "

        )
