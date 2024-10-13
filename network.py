import torch
import torch.nn as nn
from torchsummary import summary
from torchvision.transforms.v2 import Resize, InterpolationMode


def weight_init(layer):
    with torch.no_grad():
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
            nn.init.normal_(layer.weight.data, 0.0, 0.02)
            if hasattr(layer, "bias") and layer.bias is not None:
                nn.init.constant_(layer.bias.data, 0.0)


class ResizeConv(nn.Module):
    def __init__(
        self, in_channels, out_channels, size, kernel=2, stride=2, padding=0
    ) -> None:
        super().__init__()

        self.resize = Resize(size, InterpolationMode.NEAREST)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel, stride, padding)

    def forward(self, X):
        return self.conv(self.resize(X))


class ResidualBlock(nn.Module):
    def __init__(
        self, channels, use_bias=True, out_norm=True, dropout=0, inst_norm=True
    ):
        super().__init__()

        self.net = nn.Sequential()

        self.net.append(
            nn.Conv2d(
                channels,
                channels,
                kernel_size=4,
                padding=2,
                bias=use_bias,
                padding_mode="reflect",
            )
        )
        if inst_norm:
            self.net.append(nn.InstanceNorm2d(channels))
        else:
            self.net.append(nn.BatchNorm2d(channels))
        self.net.append(nn.LeakyReLU(0.2))
        if dropout > 0:
            self.net.append(nn.Dropout2d(dropout))
        self.net.append(
            nn.Conv2d(
                channels,
                channels,
                kernel_size=4,
                padding=1,
                bias=use_bias,
                padding_mode="reflect",
            )
        )
        if out_norm:
            if inst_norm:
                self.net.append(nn.InstanceNorm2d(channels))
            else:
                self.net.append(nn.BatchNorm2d(channels))

    def forward(self, x):
        return x + self.net(x)


class UpDownBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel=2,
        stride=2,
        padding=1,
        output_padding=1,
        down=True,
        dropout=0,
        use_bias=True,
        use_activation=True,
        use_norm=False,
        res_layers=0,
        size=(256, 256),
        resize=True,
        transp=True,
        inst_norm=True,
    ) -> None:
        super().__init__()

        self.net = nn.Sequential()

        if down or not resize:
            self.net.append(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel,
                    stride,
                    padding,
                    padding_mode="reflect",
                    bias=use_bias,
                )
            )
        else:
            if transp:
                self.net.append(
                    nn.ConvTranspose2d(
                        in_channels,
                        out_channels,
                        kernel,
                        stride,
                        padding,
                        output_padding=output_padding,
                        bias=use_bias,
                    )
                )
            else:
                self.net.append(
                    ResizeConv(in_channels, out_channels, size, kernel, stride, padding)
                )

        if use_norm:
            if inst_norm:
                self.net.append(nn.InstanceNorm2d(out_channels))
            else:
                self.net.append(nn.BatchNorm2d(out_channels))
        if use_activation:
            self.net.append(nn.LeakyReLU(0.2))
        if dropout > 0:
            self.net.append(nn.Dropout2d(dropout))
        for _ in range(res_layers):
            self.net.append(ResidualBlock(out_channels))

    def forward(self, X):
        return self.net(X)


class Discriminator(nn.Module):
    def __init__(
        self, layers=3, base_channels=64, res_layers=1, dropout=0.5, inst_norm=True
    ) -> None:
        super().__init__()

        self.net = nn.Sequential()

        self.net.append(
            UpDownBlock(3, base_channels, 7, 1, use_norm=False, inst_norm=inst_norm)
        )
        self.net.append(
            UpDownBlock(
                base_channels,
                base_channels,
                dropout=dropout,
                use_norm=False,
                inst_norm=inst_norm,
            )
        )
        mid_channels = base_channels
        for _ in range(layers):
            self.net.append(
                UpDownBlock(
                    mid_channels, mid_channels * 2, dropout=dropout, inst_norm=inst_norm
                )
            )
            mid_channels *= 2

        for i in range(res_layers):
            self.net.append(
                ResidualBlock(
                    mid_channels, out_norm=i < (res_layers - 1), inst_norm=inst_norm
                )
            )
        self.net.append(nn.Conv2d(mid_channels, 1, 6, 2))
        self.net.append(nn.ReLU())
        self.net.append(nn.Conv2d(1, 1, 4, 2))
        # self.net.append(nn.AdaptiveAvgPool2d(3))
        # self.net.append(nn.Flatten())
        # self.net.append(nn.Linear(9, 1))
        # if use_sigmoid:
        # self.net.append(nn.Sigmoid())

        self.apply(weight_init)

    def forward(self, X):
        return self.net(X)


class Generator(nn.Module):
    def __init__(
        self,
        in_channels=3,
        base_channels=64,
        levels=2,
        res_layers=3,
        kernel=2,
        ud_res_layers=1,
        res_dropout=0,
        inst_norm=True,
    ) -> None:
        super().__init__()

        self.net = nn.Sequential()

        self.net.append(
            UpDownBlock(
                in_channels,
                base_channels,
                7,
                1,
                3,
                res_layers=ud_res_layers,
                use_norm=False,
                inst_norm=inst_norm,
            )
        )

        tmp_channels = base_channels
        for _ in range(levels):
            self.net.append(
                UpDownBlock(
                    tmp_channels,
                    2 * tmp_channels,
                    kernel,
                    2,
                    1,
                    res_layers=ud_res_layers,
                    inst_norm=inst_norm,
                )
            )
            tmp_channels *= 2

        for _ in range(res_layers):
            self.net.append(
                ResidualBlock(tmp_channels, dropout=res_dropout, inst_norm=inst_norm)
            )

        base_size = 256 / (2**levels)

        for i in range(levels):
            new_size = int(base_size * 2 ** (i + 1))
            self.net.append(
                UpDownBlock(
                    tmp_channels,
                    tmp_channels // 2,
                    kernel,
                    2,
                    1,
                    0,
                    down=False,
                    res_layers=ud_res_layers,
                    size=(new_size, new_size),
                    inst_norm=inst_norm,
                )
            )
            tmp_channels = tmp_channels // 2

        self.net.append(
            UpDownBlock(
                base_channels,
                in_channels,
                3,
                1,
                4,
                0,
                down=False,
                use_activation=True,
                resize=False,
                res_layers=ud_res_layers,
                inst_norm=inst_norm,
            )
        )
        self.net.append(nn.Conv2d(in_channels, in_channels, 7, 1, 0))
        self.net.append(nn.Tanh())

        self.apply(weight_init)

    def forward(self, X):
        return self.net(X)


class UNet(nn.Module):
    def __init__(
        self,
        inner_net=None,
        in_channels=3,
        base_channels=64,
        use_skip=True,
        kernel=4,
        stride=2,
        padding=1,
        output_padding=0,
        res_layers=1,
        dropout=0,
        in_norm=False,
        out_norm=False,
        use_tanh=False,
        out_act=True,
        inst_norm=True,
    ) -> None:
        super().__init__()
        self.use_skip = use_skip

        self.net = nn.Sequential()

        # Downsample
        self.net.append(
            UpDownBlock(
                in_channels,
                base_channels,
                kernel,
                stride,
                padding,
                output_padding,
                use_norm=in_norm,
                inst_norm=inst_norm,
            )
        )
        for _ in range(res_layers):
            self.net.append(ResidualBlock(base_channels, inst_norm=inst_norm))

        if inner_net is not None:
            self.net.append(inner_net)

        # Upsample
        for _ in range(res_layers):
            self.net.append(
                ResidualBlock(base_channels, inst_norm=inst_norm, dropout=dropout)
            )

        self.net.append(
            UpDownBlock(
                base_channels,
                in_channels,
                kernel,
                stride,
                padding,
                output_padding,
                down=False,
                use_activation=out_act,
                use_norm=out_norm,
                inst_norm=inst_norm,
            )
        )

        self.last = nn.Sequential()
        self.last.append(
            nn.Conv2d(in_channels * 2, in_channels, 3, 1, 1, padding_mode="reflect")
        )

        if use_tanh:
            self.net.append(nn.Tanh())
        else:
            self.last.append(nn.LeakyReLU(0.2))

        self.apply(weight_init)

    def forward(self, X):
        if self.use_skip:
            # Concatenate channels
            return self.last(torch.cat([X, self.net(X)], 1))
        else:
            return self.net(X)


def UNet_generator(
    in_channels=3,
    base_channels=64,
    layers=3,
    center_res=1,
    res_layers=1,
    dropout=0.5,
    inst_norm=True,
):
    tmp = nn.Sequential()
    for _ in range(center_res):
        tmp.append(
            ResidualBlock(
                base_channels * (2**layers), inst_norm=inst_norm, dropout=dropout
            )
        )
    for i in range(layers):
        tmp = UNet(
            tmp,
            base_channels * (2 ** (layers - i - 1)),
            base_channels * (2 ** (layers - i)),
            res_layers=res_layers,
            inst_norm=inst_norm,
            dropout=dropout,
        )
    return UNet(
        tmp,
        in_channels,
        base_channels,
        use_skip=False,
        use_tanh=True,
        in_norm=False,
        out_norm=False,
        out_act=False,
        res_layers=res_layers,
        inst_norm=inst_norm,
    )


if __name__ == "__main__":
    from AutoConfig import args_from_YAML

    args = args_from_YAML(r"D:\Atom\MSDS\DTSA5511\CycleGAN\config.yaml")

    # net = Discriminator(**args.dis.get_kwargs())
    net = Generator(**args.gen.get_kwargs())
    # net = ResidualBlock(3)

    # net = UNet_generator(**args.unet.get_kwargs())

    summary(net, torch.ones((1, 3, 256, 256)), device="cpu", depth=20)
