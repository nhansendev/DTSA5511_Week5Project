import os
import torch
import torch.nn as nn
from itertools import chain

from data_handling import ImagePool
from network import UNet_generator, Generator, Discriminator
from constants import RNG
from utils import set_req_grad
from plotting import plot_example, plot_hist


class Trainer:
    def __init__(
        self,
        working_dir,
        examplesA,
        examplesB,
        args,
    ) -> None:
        self.working_dir = working_dir
        self.examplesA = examplesA
        self.examplesB = examplesB

        self.total_steps = 0
        self.warmup_steps = args.trainer.warmup_steps

        self.lambda_A = args.trainer.lambda_A
        self.lambda_B = args.trainer.lambda_B
        self.lambda_idt = args.trainer.lambda_idt
        self.label_noise = args.trainer.label_noise
        self.swap_prob = args.trainer.swap_prob
        self.instance_noise = args.trainer.instance_noise

        self.pool_A = ImagePool(args.trainer.pool_size)
        self.pool_B = ImagePool(args.trainer.pool_size)

        if args.trainer.use_unet:
            self.gen_A = UNet_generator(**args.unet.get_kwargs()).cuda()
            self.gen_B = UNet_generator(**args.unet.get_kwargs()).cuda()
        else:
            self.gen_A = Generator(**args.gen.get_kwargs()).cuda()
            self.gen_B = Generator(**args.gen.get_kwargs()).cuda()

        self.dis_A = Discriminator(**args.dis.get_kwargs()).cuda()
        self.dis_B = Discriminator(**args.dis.get_kwargs()).cuda()

        self.optG = torch.optim.AdamW(
            chain(self.gen_A.parameters(), self.gen_B.parameters()),
            **args.opt_G.get_kwargs(),
        )
        self.optD = torch.optim.SGD(
            chain(self.dis_A.parameters(), self.dis_B.parameters()),
            **args.opt_D.get_kwargs(),
        )

        # self.sched_G = torch.
        self.sched_D = torch.optim.lr_scheduler.ExponentialLR(self.optD, 0.999)
        self.lr_hist_D = []

        self.L1 = nn.L1Loss()
        if args.trainer.use_MSE:
            self.L2 = nn.MSELoss()
        else:
            self.L2 = nn.BCEWithLogitsLoss()

        self.pred_shape = self.dis_A(
            torch.zeros((args.batch_size, 3, 256, 256)).cuda()
        ).shape
        self.comp_true = torch.ones(self.pred_shape).cuda()
        self.comp_false = torch.zeros(self.pred_shape).cuda()

        self.loss_hist_GA_dis = []
        self.loss_hist_GA_cyc = []
        self.loss_hist_GA_idt = []
        self.loss_hist_GB_dis = []
        self.loss_hist_GB_cyc = []
        self.loss_hist_GB_idt = []
        self.loss_hist_DA = []
        self.loss_hist_DB = []

    def gen_labels(self):
        # "Soft" labels
        labels_true = (
            torch.ones(self.pred_shape)
            + torch.rand(self.pred_shape) * self.label_noise
            - self.label_noise
        ).cuda()
        labels_false = (
            torch.zeros(self.pred_shape)
            + torch.rand(self.pred_shape) * self.label_noise
        ).cuda()
        if RNG.uniform() < self.swap_prob:
            # Swap labels randomly
            return labels_false, labels_true
        else:
            return labels_true, labels_false

    def step(self, real_A, real_B):
        self.optG.zero_grad()

        fake_A = self.gen_A(real_B)
        fake_B = self.gen_B(real_A)

        cyc_A = self.gen_A(fake_B)
        cyc_B = self.gen_B(fake_A)

        idt_A = self.gen_A(real_A)
        idt_B = self.gen_B(real_B)

        # Training Warmup
        if self.total_steps >= self.warmup_steps:
            with torch.no_grad():
                loss_GA_dis = self.L2(self.dis_A(fake_A), self.comp_true)
                loss_GB_dis = self.L2(self.dis_B(fake_B), self.comp_true)

            self.loss_hist_GA_dis.append(loss_GA_dis.item())
            self.loss_hist_GB_dis.append(loss_GB_dis.item())
        else:
            # Ignore discriminator and focus on recreating the images
            loss_GA_dis = 0
            loss_GB_dis = 0
            self.loss_hist_GA_dis.append(0)
            self.loss_hist_GB_dis.append(0)

        loss_GA_cyc = self.L1(cyc_A, real_A) * self.lambda_A
        loss_GB_cyc = self.L1(cyc_B, real_B) * self.lambda_B

        loss_GA_idt = self.L1(idt_A, real_A) * self.lambda_A * self.lambda_idt
        loss_GB_idt = self.L1(idt_B, real_B) * self.lambda_B * self.lambda_idt

        self.loss_hist_GA_cyc.append(loss_GA_cyc.item())
        self.loss_hist_GA_idt.append(loss_GA_idt.item())
        self.loss_hist_GB_cyc.append(loss_GB_cyc.item())
        self.loss_hist_GB_idt.append(loss_GB_idt.item())

        loss_tot_G = (
            loss_GA_dis
            + loss_GA_cyc
            + loss_GA_idt
            + loss_GB_dis
            + loss_GB_cyc
            + loss_GB_idt
        )

        loss_tot_G.backward()
        self.optG.step()

        if self.total_steps >= self.warmup_steps:
            if self.instance_noise > 0:
                noise = (
                    torch.rand_like(fake_A).cuda() * self.instance_noise
                    - self.instance_noise / 2
                )
            else:
                noise = 0

            self.optD.zero_grad()
            label_true, label_false = self.gen_labels()
            # Only update on real or fake, not both
            if self.total_steps % 2 == 0:
                d_fake_A = self.pool_A(fake_A.detach())
                d_fake_B = self.pool_B(fake_B.detach())
                loss_DA = self.L2(self.dis_A(d_fake_A + noise), label_false)
                loss_DB = self.L2(self.dis_B(d_fake_B + noise), label_false)
            else:
                loss_DA = self.L2(self.dis_A(real_A + noise), label_true)
                loss_DB = self.L2(self.dis_B(real_B + noise), label_true)

            self.loss_hist_DA.append(loss_DA.item())
            self.loss_hist_DB.append(loss_DB.item())

            loss_tot_D = (loss_DA + loss_DB) / 2
            loss_tot_D.backward()
            self.optD.step()

            # self.sched_D.step()
            # self.lr_hist_D.append(self.sched_D.get_last_lr()[0])
        else:
            self.loss_hist_DA.append(0)
            self.loss_hist_DB.append(0)

        self.total_steps += 1

    def save_models(self):
        checkpoint = {
            "gen_A": self.gen_A.state_dict(),
            "gen_B": self.gen_B.state_dict(),
            "dis_A": self.dis_A.state_dict(),
            "dis_B": self.dis_B.state_dict(),
            "optG": self.optG.state_dict(),
            "optD": self.optD.state_dict(),
        }
        torch.save(checkpoint, os.path.join(self.working_dir, "checkpoint.pt"))

    def load_models(self, folder):
        checkpoint = torch.load(os.path.join(folder, "checkpoint.pt"))

        self.gen_A.load_state_dict(checkpoint["gen_A"])
        self.gen_B.load_state_dict(checkpoint["gen_B"])
        self.dis_A.load_state_dict(checkpoint["dis_A"])
        self.dis_B.load_state_dict(checkpoint["dis_B"])
        self.optG.load_state_dict(checkpoint["optG"])
        self.optD.load_state_dict(checkpoint["optD"])

    def checkpoint(self):
        plot_example(self)
        plot_hist(self)
