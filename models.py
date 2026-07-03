import torch
from torch import nn


class IdentityAttention(nn.Module):
    def forward(self, x):
        return x


class SEAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        hidden_dim = max(channels // reduction, 4)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, hidden_dim, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        weights = self.fc(self.pool(x))
        return x * weights


class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        hidden_dim = max(channels // reduction, 4)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, hidden_dim, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, channels, kernel_size=1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.mlp(self.avg_pool(x))
        max_out = self.mlp(self.max_pool(x))
        return x * self.sigmoid(avg_out + max_out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = x.mean(dim=1, keepdim=True)
        max_out, _ = x.max(dim=1, keepdim=True)
        attention = self.sigmoid(self.conv(torch.cat((avg_out, max_out), dim=1)))
        return x * attention


class CBAttention(nn.Module):
    def __init__(self, channels, reduction=16, spatial_kernel_size=7):
        super().__init__()
        self.channel_attention = ChannelAttention(channels, reduction=reduction)
        self.spatial_attention = SpatialAttention(kernel_size=spatial_kernel_size)

    def forward(self, x):
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x


def _build_attention(name, channels, reduction=16):
    name = name.lower()
    if name == "none":
        return IdentityAttention()
    if name == "se":
        return SEAttention(channels, reduction=reduction)
    if name == "cbam":
        return CBAttention(channels, reduction=reduction)
    raise ValueError(f"Unsupported attention type: {name}")


class SRCNN(nn.Module):
    def __init__(
        self,
        num_channels=1,
        attention_type="none",
        attention_position="after_conv2",
        reduction=16,
        kernel_sizes=(9, 5, 5),
    ):
        super().__init__()
        if len(kernel_sizes) != 3:
            raise ValueError(f"Expected three kernel sizes, got: {kernel_sizes}")
        k1, k2, k3 = kernel_sizes
        self.conv1 = nn.Conv2d(num_channels, 64, kernel_size=k1, padding=k1 // 2)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=k2, padding=k2 // 2)
        self.conv3 = nn.Conv2d(32, num_channels, kernel_size=k3, padding=k3 // 2)
        self.relu = nn.ReLU(inplace=True)
        self.attention_position = attention_position
        self.attn1 = _build_attention(attention_type, 64, reduction=reduction)
        self.attn2 = _build_attention(attention_type, 32, reduction=reduction)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        if self.attention_position == "after_conv1":
            x = self.attn1(x)

        x = self.relu(self.conv2(x))
        if self.attention_position == "after_conv2":
            x = self.attn2(x)

        x = self.conv3(x)
        return x


def create_model(
    model_name="srcnn_baseline",
    num_channels=1,
    attention_type="none",
    attention_position="after_conv2",
    kernel_sizes=(9, 5, 5),
):
    model_name = model_name.lower()
    if model_name == "srcnn_baseline":
        return SRCNN(
            num_channels=num_channels,
            attention_type="none",
            attention_position=attention_position,
            kernel_sizes=kernel_sizes,
        )
    if model_name == "srcnn_attention":
        return SRCNN(
            num_channels=num_channels,
            attention_type=attention_type,
            attention_position=attention_position,
            kernel_sizes=kernel_sizes,
        )
    raise ValueError(f"Unknown model name: {model_name}")
