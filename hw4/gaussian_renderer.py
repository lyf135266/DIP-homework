import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple
from dataclasses import dataclass
import numpy as np
import cv2


class GaussianRenderer(nn.Module):
    def __init__(self, image_height: int, image_width: int):
        super().__init__()
        self.H = image_height
        self.W = image_width

        # Pre-compute pixel coordinates grid
        y, x = torch.meshgrid(
            torch.arange(image_height, dtype=torch.float32),
            torch.arange(image_width, dtype=torch.float32),
            indexing='ij'
        )
        # Shape: (H, W, 2)
        self.register_buffer('pixels', torch.stack([x, y], dim=-1))
    def compute_projection(self, means3D, covs3d, K, R, t):
        N = means3D.shape[0]

        cam_points = means3D @ R.T + t.unsqueeze(0)  # (N, 3)
        depths = cam_points[:, 2].clamp(min=1.)
        screen_points = cam_points @ K.T
        means2D = screen_points[..., :2] / screen_points[..., 2:3]

        J_proj = torch.zeros((N, 2, 3), device=means3D.device)
        J_proj[:, 0, 0] = 1.0 / cam_points[:, 2]
        J_proj[:, 1, 1] = 1.0 / cam_points[:, 2]
        J_proj[:, 0, 2] = -cam_points[:, 0] / (cam_points[:, 2] ** 2)
        J_proj[:, 1, 2] = -cam_points[:, 1] / (cam_points[:, 2] ** 2)

        covs_cam = torch.bmm(R.unsqueeze(0).expand(N, -1, -1), torch.bmm(covs3d, R.T.unsqueeze(0).expand(N, -1, -1)))
        covs2D = torch.bmm(J_proj, torch.bmm(covs_cam, J_proj.permute(0, 2, 1)))

        return means2D, covs2D, depths

    def compute_gaussian_values(self, means2D, covs2D, pixels):
        N = means2D.shape[0]
        H, W = pixels.shape[:2]

        dx = pixels.unsqueeze(0) - means2D.reshape(N, 1, 1, 2)
        eps = 1e-4
        covs2D = covs2D + eps * torch.eye(2, device=covs2D.device).unsqueeze(0)

        inv_covs = torch.inverse(covs2D)
        det_covs = torch.det(covs2D).clamp(min=1e-6)

        dx = dx.unsqueeze(-1)
        mahalanobis_dist = torch.matmul(dx.transpose(-2, -1), torch.matmul(inv_covs.unsqueeze(1).unsqueeze(1), dx))
        mahalanobis_dist = mahalanobis_dist.squeeze(-1).squeeze(-1)

        gaussian = torch.exp(-0.5 * mahalanobis_dist) / (2 * np.pi * torch.sqrt(det_covs)).view(N, 1, 1)
        return gaussian

    def forward(self, means3D, covs3d, colors, opacities, K, R, t):
        N = means3D.shape[0]
        means2D, covs2D, depths = self.compute_projection(means3D, covs3d, K, R, t)
        valid_mask = (depths > 1.) & (depths < 50.0)

        indices = torch.argsort(depths, dim=0, descending=False)
        means2D = means2D[indices]
        covs2D = covs2D[indices]
        colors = colors[indices]
        opacities = opacities[indices]
        valid_mask = valid_mask[indices]

        gaussian_values = self.compute_gaussian_values(means2D, covs2D, self.pixels)
        gaussian_values = gaussian_values * valid_mask.view(N, 1, 1)

        alphas = opacities.view(N, 1, 1) * gaussian_values
        colors = colors.view(N, 3, 1, 1).expand(-1, -1, self.H, self.W).permute(0, 2, 3, 1)

        weights = alphas * torch.cumprod(1 - alphas + 1e-4, dim=0).roll(1, dims=0)
        weights[0] = alphas[0]

        rendered = (weights.unsqueeze(-1) * colors).sum(dim=0)
        return rendered
