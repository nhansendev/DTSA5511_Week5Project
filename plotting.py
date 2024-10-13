import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

from utils import MAV


def plot_img_tensor(data, r=2, c=6, w=12, h=4):
    if isinstance(data, np.ndarray):
        imgs = data
    else:
        tmp = torch.permute((data[: r * c] + 1) * 127.5, (0, 2, 3, 1))
        imgs = tmp.cpu().detach().numpy().astype(int)

    fig, ax = plt.subplots(r, c)
    axs = fig.axes
    fig.set_size_inches(w, h)

    for i, a in enumerate(axs):
        a.imshow(imgs[i])
        a.axis("off")

    plt.tight_layout()
    plt.show()


def plot_hist(
    trainer,
    av_length=50,
    width=12,
    height=8,
    save=True,
    show=False,
    log_scale=True,
    ylim=[0.01, 10],
):
    def _plot(hists, colors, X=None):
        if X is None:
            X = range(len(hists[0]))

        maH = [MAV(h, av_length) for h in hists]

        for i in range(len(hists)):
            axs[0].plot(
                X,
                maH[i],
                colors[i],
                path_effects=[pe.Stroke(linewidth=3, foreground="k"), pe.Normal()],
            )

        return maH

    def _plot2(hists, maH, colors, X=None):
        if X is None:
            X = range(len(hists[0]))

        for i, h in enumerate(hists):
            if len(h) > 0:
                axs[0].plot(X, h, ".", color=colors[i], alpha=0.1, zorder=-10)

        axs[0].hlines(
            [h[-1] for h in maH if len(h) > 0],
            0,
            max(X),
            color="k",
            linestyles="dashed",
        )

        for i, h in enumerate(maH):
            if len(h) > 0:
                axs[0].plot([0, max(X)], [h[-1], h[-1]], ".", color=colors[i])

    fig, ax = plt.subplots(1, 1)
    axs = fig.axes
    fig.set_size_inches(width, height)

    hists = [
        trainer.loss_hist_GA_dis,
        trainer.loss_hist_GB_dis,
        trainer.loss_hist_GA_cyc,
        trainer.loss_hist_GB_cyc,
        trainer.loss_hist_GA_idt,
        trainer.loss_hist_GB_idt,
    ]

    hists2 = [
        trainer.loss_hist_DA,
        trainer.loss_hist_DB,
    ]

    colors = ["c", "r", "tab:orange", "tab:gray", "tab:pink", "tab:brown"]
    colors2 = ["g", "m"]

    maH = _plot(hists, colors)
    if len(hists2[0]) > 0:
        maH2 = _plot(hists2, colors2)

    _plot2(hists, maH, colors)
    if len(hists2[0]) > 0:
        _plot2(hists2, maH2, colors2)

    axs[0].set_ylabel("Loss")
    axs[0].set_xlabel("Step")
    if log_scale:
        axs[0].set_yscale("log")
    axs[0].set_ylim(ylim)
    axs[0].grid()
    axs[0].legend(["GA_D", "GB_D", "GA_C", "GB_C", "GA_I", "GB_I", "DA", "DB"])

    ax1 = axs[0].twinx()
    ax1.plot(trainer.lr_hist_D, "-", color="tab:olive", alpha=0.5)
    ax1.set_yscale("log")
    ax1.set_ylabel("Learning Rate")

    if save:
        plt.savefig(os.path.join(trainer.working_dir, "0hist.png"))
    if show:
        plt.show()
    else:
        plt.close()


def plot_example(trainer, s=1.5):
    def _conv(data):
        return (
            torch.permute((data + 1) * 127.5, (0, 2, 3, 1))
            .cpu()
            .detach()
            .numpy()
            .astype(int)
        )

    def _flat(data):
        target_axis = tuple(range(1, len(data.shape)))  # Ignore batch dimension
        tmp = torch.mean(data, target_axis).cpu().detach().numpy().flatten().tolist()
        return [round(t, 2) for t in tmp]

    def _plot(data, axs, title=None, label=None):
        if title is not None:
            axs.set_title(title)
        axs.imshow(data)
        axs.set_xticks([])
        axs.set_yticks([])
        axs.set_xticklabels([])
        axs.set_yticklabels([])
        if label is not None:
            axs.set_ylabel(label)

    N = trainer.examplesA.shape[0]

    w = 2 * N * s
    h = 4 * s + 0.5

    with torch.no_grad():
        # with torch.autocast(device_type="cuda"):
        real_A = trainer.examplesA
        real_B = trainer.examplesB

        gen_A = trainer.gen_A
        gen_B = trainer.gen_B

        gen_A.eval()
        gen_B.eval()
        trainer.dis_A.eval()
        trainer.dis_B.eval()

        fake_A = gen_A(real_B)
        fake_B = gen_B(real_A)

        rec_A = gen_A(fake_B)
        rec_B = gen_B(fake_A)

        idt_A = gen_A(real_A)
        idt_B = gen_B(real_B)

        rA = _flat(trainer.dis_A(real_A))
        fA = _flat(trainer.dis_A(fake_A))

        rB = _flat(trainer.dis_B(real_B))
        fB = _flat(trainer.dis_B(fake_B))

        gen_A.train()
        gen_B.train()
        trainer.dis_A.train()
        trainer.dis_B.train()

    fig, axs = plt.subplots(4, N * 2)
    fig.set_size_inches(w, h)

    # Show the discriminator outputs
    # plt.suptitle(f"RA {rA} | RB {rB}\nFA {fA} | FB {fB}")
    A_titles = [f"{r:.0%} | {1-f:.0%}" for r, f in list(zip(rA, fA))]
    B_titles = [f"{r:.0%} | {1-f:.0%}" for r, f in list(zip(rB, fB))]

    labels = ["real", "fake", "cycle", "idt"]
    pairs = [[real_A, real_B], [fake_B, fake_A], [rec_A, rec_B], [idt_A, idt_B]]

    for j, (p1, p2) in enumerate(pairs):
        imgs = _conv(p1)
        for i in range(N):
            title = A_titles[i] if j == 0 else None
            label = labels[j] if i == 0 else None
            _plot(imgs[i], axs[j, i], title, label)

        imgs = _conv(p2)
        for i in range(N):
            title = B_titles[i] if j == 0 else None
            _plot(imgs[i], axs[j, i + N], title)

    plt.tight_layout()

    filecount = len([f for f in os.listdir(trainer.working_dir) if "ex_" in f])
    plt.savefig(os.path.join(trainer.working_dir, f"ex_{filecount}.png"))
    plt.close()
