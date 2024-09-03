import contextlib
import time
import datetime
import math

import torch
import torch.nn as nn
from torch.nn import functional as F

from dassl.data import DataManager
from dassl.engine import TRAINER_REGISTRY, TrainerXU, SimpleNet
from dassl.optim import build_optimizer, build_lr_scheduler
from dassl.data.transforms import build_transform
from dassl.utils import count_num_param

@contextlib.contextmanager
def freeze_models_params(models):
    try:
        for model in models:
            for param in model.parameters():
                param.requires_grad_(False)
        yield
    finally:
        for model in models:
            for param in model.parameters():
                param.requires_grad_(True)

class DGWMClassifier(nn.Module):
    def __init__(self, num_features, num_classes, LR=False, noise=False):
        super().__init__()
        self.w = nn.Parameter(torch.Tensor(num_classes, num_features))

        if LR:
            self.h1 = nn.Linear(num_features, num_features)
            self.h2 = nn.Linear(num_features, num_classes)

        self.p1 = nn.Linear(num_features, num_features//2)
        self.p2 = nn.Linear(num_features//2, num_features//4)
        self.p3 = nn.Linear(num_features//4, num_features//8)

        self.p4 = nn.Linear(num_features//4, num_features//2)
        self.p5 = nn.Linear(num_features//2, num_features)

        stdv = 1./math.sqrt(self.w.size(1))
        self.w.data.uniform_(-stdv, stdv)

    def forward(self, x, LR=False, noise=False):
        if LR:
            x_mean = x.mean(0).unsqueeze(0)

            x_mean = torch.relu(self.p1(x_mean)) # 256
            x_mean = torch.relu(self.p2(x_mean)) # 128
            x_mean = torch.sigmoid(self.p3(x_mean)) # 64

            if noise:
                noise = torch.randn(1, int(x.shape[1]/8)).to(x.device) 
                x_mean = torch.cat((x_mean, noise), dim=1) # 128      
            else:
                x_mean = torch.cat((x_mean, torch.zeros(1, int(x.shape[1]/8)).to(x.device)), dim=1) # 128

            x_mean = torch.relu(self.p4(x_mean)) # 256
            x_mean = torch.relu(self.p5(x_mean)) # 512

            x1 = self.h1(x_mean)
            x2 = self.h2(x_mean)

            w_mask = torch.sigmoid(torch.matmul(x2.t(), x1))
            self.w_new = self.w * w_mask
            
            return torch.matmul(x, self.w_new.t())                        
        else:
            return torch.matmul(x, self.w.t()) 
 

@TRAINER_REGISTRY.register()
class DGWM(TrainerXU):
    """
    Domain-Guided Weight Modulation for Semi-Supervised Domain Generalization

    """
    def __init__(self, cfg):
        super().__init__(cfg)
        # Confidence threshold
        self.conf_thre = cfg.TRAINER.DGWM.CONF_THRE

        norm_mean = None
        norm_std = None

        if "normalize" in cfg.INPUT.TRANSFORMS:
            norm_mean = cfg.INPUT.PIXEL_MEAN
            norm_std = cfg.INPUT.PIXEL_STD

        self.apply_aug = cfg.TRAINER.DGWM.APPLY_AUG

    def check_cfg(self, cfg):
        assert len(cfg.TRAINER.DGWM.STRONG_TRANSFORMS) > 0
        assert cfg.DATALOADER.TRAIN_X.SAMPLER == "SeqDomainSampler"
        assert cfg.DATALOADER.TRAIN_U.SAME_AS_X

    def build_data_loader(self):
        cfg = self.cfg
        tfm_train = build_transform(cfg, is_train=True)
        custom_tfm_train = [tfm_train]
        choices = cfg.TRAINER.DGWM.STRONG_TRANSFORMS
        tfm_train_strong = build_transform(cfg, is_train=True, choices=choices)
        custom_tfm_train += [tfm_train_strong]
        dm = DataManager(self.cfg, custom_tfm_train=custom_tfm_train)
        self.train_loader_x = dm.train_loader_x
        self.train_loader_u = dm.train_loader_u
        self.val_loader = dm.val_loader
        self.test_loader = dm.test_loader
        self.num_classes = dm.num_classes
        self.num_source_domains = dm.num_source_domains
        self.lab2cname = dm.lab2cname

    def build_model(self):
        cfg = self.cfg

        print("Building G")
        self.G = SimpleNet(cfg, cfg.MODEL, 0)  # n_class=0: only produce features
        self.G.to(self.device)
        print("# params: {:,}".format(count_num_param(self.G)))
        self.optim_G = build_optimizer(self.G, cfg.OPTIM)
        self.sched_G = build_lr_scheduler(self.optim_G, cfg.OPTIM)
        self.register_model("G", self.G, self.optim_G, self.sched_G)

        print("Building C")
        self.C = DGWMClassifier(self.G.fdim, self.num_classes, LR=True, noise=False)
        self.C.to(self.device)
        print("# params: {:,}".format(count_num_param(self.C)))
        self.optim_C = build_optimizer(self.C, cfg.TRAINER.DGWM.C_OPTIM)
        self.sched_C = build_lr_scheduler(self.optim_C, cfg.TRAINER.DGWM.C_OPTIM)
        self.register_model("C", self.C, self.optim_C, self.sched_C)

    def assess_y_pred_quality(self, y_pred, y_true, mask):
        n_masked_correct = (y_pred.eq(y_true).float() * mask).sum()
        acc_thre = n_masked_correct / (mask.sum() + 1e-5)  # accuracy after threshold
        acc_raw = y_pred.eq(y_true).sum() / y_pred.numel()  # raw accuracy
        keep_rate = mask.sum() / mask.numel()
        output = {"acc_thre": acc_thre, "acc_raw": acc_raw, "keep_rate": keep_rate}
        return output

    def forward_backward(self, batch_x, batch_u):
        parsed_batch = self.parse_batch_train(batch_x, batch_u)

        x0 = parsed_batch["x0"]
        x = parsed_batch["x"]
        x_aug = parsed_batch["x_aug"]
        y_x_true = parsed_batch["y_x_true"]

        u0 = parsed_batch["u0"]
        u = parsed_batch["u"]
        u_aug = parsed_batch["u_aug"]
        y_u_true = parsed_batch["y_u_true"]  # tensor

        K = self.num_source_domains
        # NOTE: If num_source_domains=1, we split a batch into two halves
        K = 2 if K == 1 else K

        ####################
        # Generate pseudo labels
        ####################
        with torch.no_grad():
            p_xu = []
            for k in range(K):
                x_k = x[k]
                u_k = u[k]
                xu_k = torch.cat([x_k, u_k], 0)
                f_xu_k = self.G(xu_k)
                z_xu_k = self.C(f_xu_k, LR=True)
                p_xu_k = F.softmax(z_xu_k, 1)
                p_xu.append(p_xu_k)
            p_xu = torch.cat(p_xu, 0)

            p_xu_maxval, y_xu_pred = p_xu.max(1)
            mask_xu = (p_xu_maxval >= self.conf_thre).float()

            y_xu_pred = y_xu_pred.chunk(K)
            mask_xu = mask_xu.chunk(K)

            # Calculate pseudo-label's accuracy
            y_u_pred = []
            mask_u = []
            for y_xu_k_pred, mask_xu_k in zip(y_xu_pred, mask_xu):
                y_u_pred.append(
                    y_xu_k_pred.chunk(2)[1]
                )  # only take the 2nd half (unlabeled data)
                mask_u.append(mask_xu_k.chunk(2)[1])
            y_u_pred = torch.cat(y_u_pred, 0)
            mask_u = torch.cat(mask_u, 0)
            y_u_pred_stats = self.assess_y_pred_quality(y_u_pred, y_u_true, mask_u)
        
        ####################
        # Supervised loss
        ####################
        loss_x = 0
        for k in range(K):
            x_k = x[k]
            y_x_k_true = y_x_true[k]
            z_x_k = self.C(self.G(x_k), LR=True, noise=True)
            loss_x += F.cross_entropy(z_x_k, y_x_k_true)

        ####################
        # Unsupervised loss
        ####################
        loss_u_aug = 0
        loss_u_sty = 0
        for k in range(K):
            y_xu_k_pred = y_xu_pred[k]
            mask_xu_k = mask_xu[k]

            # Compute loss for strongly augmented data
            if self.apply_aug:
                x_k_aug = x_aug[k]
                u_k_aug = u_aug[k]
                xu_k_aug = torch.cat([x_k_aug, u_k_aug], 0)
                f_xu_k_aug = self.G(xu_k_aug)
                z_xu_k_aug = self.C(f_xu_k_aug, LR=True, noise=True)
                loss = F.cross_entropy(z_xu_k_aug, y_xu_k_pred, reduction="none")
                loss = (loss * mask_xu_k).mean()
                loss_u_aug += loss

        loss_summary = {}

        loss_all = 0
        loss_all += loss_x
        loss_summary["loss_x"] = loss_x.item()

        if self.apply_aug:
            loss_all += loss_u_aug
            loss_summary["loss_u_aug"] = loss_u_aug.item()

        self.model_backward_and_update(loss_all)

        loss_summary["y_u_pred_acc_thre"] = y_u_pred_stats["acc_thre"]
        loss_summary["y_u_pred_acc_raw"] = y_u_pred_stats["acc_raw"]
        loss_summary["y_u_pred_keep_rate"] = y_u_pred_stats["keep_rate"]

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary

    def parse_batch_train(self, batch_x, batch_u):
        x0 = batch_x["img0"]  # no augmentation
        x = batch_x["img"]  # weak augmentation
        x_aug = batch_x["img2"]  # strong augmentation
        y_x_true = batch_x["label"]

        x0 = x0.to(self.device)
        x = x.to(self.device)
        x_aug = x_aug.to(self.device)
        y_x_true = y_x_true.to(self.device)

        u0 = batch_u["img0"]
        u = batch_u["img"]
        u_aug = batch_u["img2"]
        y_u_true = batch_u["label"]  # for evaluating pseudo labeling's accuracy only

        u0 = u0.to(self.device)
        u = u.to(self.device)
        u_aug = u_aug.to(self.device)
        y_u_true = y_u_true.to(self.device)

        # Split data into K chunks
        K = self.num_source_domains
        # NOTE: If num_source_domains=1, we split a batch into two halves
        K = 2 if K == 1 else K
        x0 = x0.chunk(K)
        x = x.chunk(K)
        x_aug = x_aug.chunk(K)
        y_x_true = y_x_true.chunk(K)
        u0 = u0.chunk(K)
        u = u.chunk(K)
        u_aug = u_aug.chunk(K)

        batch = {
            # x
            "x0": x0,
            "x": x,
            "x_aug": x_aug,
            "y_x_true": y_x_true,
            # u
            "u0": u0,
            "u": u,
            "u_aug": u_aug,
            "y_u_true": y_u_true,  # kept intact
        }
        return batch

    def model_inference(self, input):
        features = self.G(input)
        prediction = self.C(features, LR=False, noise=False)
        return prediction

    def after_train(self):
        print("Finish training")
        # Do testing
        if not self.cfg.TEST.NO_TEST:
            self.test()
        # Save model
        self.save_model(self.epoch, self.output_dir)
        # Show elapsed time
        elapsed = round(time.time() - self.time_start)
        elapsed = str(datetime.timedelta(seconds=elapsed))
        print("Elapsed: {}".format(elapsed))
        # Close writer
        self.close_writer()
