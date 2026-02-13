import torch.nn as nn

from unet import ConvBnSiLu, DecoderBlock, EncoderBlock, ResidualBottleneck


class ConditionalUnet(nn.Module):
    """
    Label-conditional U-Net.
    Keeps the original architecture but injects class embedding
    into timestep embedding.
    """

    def __init__(
        self,
        timesteps,
        time_embedding_dim,
        in_channels=3,
        out_channels=2,
        base_dim=32,
        dim_mults=(2, 4, 8, 16),
        num_classes=10,
    ):
        super().__init__()
        assert isinstance(dim_mults, (list, tuple))
        assert base_dim % 2 == 0

        self.num_classes = num_classes
        # used for classifier-free guidance unconditional path
        self.null_label = num_classes

        channels = self._cal_channels(base_dim, dim_mults)

        self.init_conv = ConvBnSiLu(in_channels, base_dim, 3, 1, 1)
        self.time_embedding = nn.Embedding(timesteps, time_embedding_dim)
        self.label_embedding = nn.Embedding(
            num_classes + 1, time_embedding_dim
        )

        self.encoder_blocks = nn.ModuleList(
            [EncoderBlock(c[0], c[1], time_embedding_dim) for c in channels]
        )
        self.decoder_blocks = nn.ModuleList(
            [
                DecoderBlock(c[1], c[0], time_embedding_dim)
                for c in channels[::-1]
            ]
        )

        self.mid_block = nn.Sequential(
            *[
                ResidualBottleneck(channels[-1][1], channels[-1][1])
                for _ in range(2)
            ],
            ResidualBottleneck(channels[-1][1], channels[-1][1] // 2),
        )

        self.final_conv = nn.Conv2d(
            in_channels=channels[0][0] // 2,
            out_channels=out_channels,
            kernel_size=1,
        )

    def forward(self, x, t=None, y=None):
        x = self.init_conv(x)
        if t is not None:
            if y is None:
                raise ValueError(
                    "labels y must be provided when timestep t is provided."
                )
            t = self.time_embedding(t) + self.label_embedding(y)
        encoder_shortcuts = []
        for encoder_block in self.encoder_blocks:
            x, x_shortcut = encoder_block(x, t)
            encoder_shortcuts.append(x_shortcut)
        x = self.mid_block(x)
        encoder_shortcuts.reverse()
        for decoder_block, shortcut in zip(
            self.decoder_blocks, encoder_shortcuts
        ):
            x = decoder_block(x, shortcut, t)
        x = self.final_conv(x)
        return x

    def _cal_channels(self, base_dim, dim_mults):
        dims = [base_dim * x for x in dim_mults]
        dims.insert(0, base_dim)
        channels = []
        for i in range(len(dims) - 1):
            channels.append((dims[i], dims[i + 1]))  # in_channel, out_channel
        return channels
