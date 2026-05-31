import torch
from torch import nn
from torchvision import models


class PerceptualFeatureExtractor(nn.Module):
    def __init__(self, layer="relu3_3", device=None):
        super().__init__()
        feature_layers = {
            "relu2_2": 8,
            "relu3_3": 15,
            "relu4_3": 24,
        }
        if layer not in feature_layers:
            raise ValueError(f"Unsupported perceptual layer: {layer}")

        weights = models.VGG19_Weights.DEFAULT
        features = models.vgg19(weights=weights).features[: feature_layers[layer] + 1]
        self.features = features.eval()
        for param in self.features.parameters():
            param.requires_grad = False

        if device is not None:
            self.to(device)

    def forward(self, x):
        return self.features(x)


class CombinedLoss(nn.Module):
    def __init__(self, loss_type="mse", perceptual_weight=0.01, perceptual_layer="relu3_3", device=None):
        super().__init__()
        self.loss_type = loss_type
        self.perceptual_weight = perceptual_weight
        self.mse_loss = nn.MSELoss()
        self.perceptual = None

        if "perceptual" in loss_type:
            self.perceptual = PerceptualFeatureExtractor(layer=perceptual_layer, device=device)
            mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1)
            self.register_buffer("imagenet_mean", mean)
            self.register_buffer("imagenet_std", std)

    def _prepare_for_vgg(self, x):
        if x.size(1) == 1:
            x = x.repeat(1, 3, 1, 1)
        return (x - self.imagenet_mean) / self.imagenet_std

    def forward(self, preds, targets):
        mse = self.mse_loss(preds, targets)
        if self.perceptual is None:
            return mse, {"mse": mse.detach().item(), "perceptual": 0.0}

        preds_vgg = self._prepare_for_vgg(preds)
        targets_vgg = self._prepare_for_vgg(targets)
        perceptual = self.mse_loss(self.perceptual(preds_vgg), self.perceptual(targets_vgg))
        total = mse + self.perceptual_weight * perceptual
        return total, {
            "mse": mse.detach().item(),
            "perceptual": perceptual.detach().item(),
            "total": total.detach().item(),
        }
