import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

# -----------------------------
# SE Block
# -----------------------------
class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        mid = max(channels // reduction, 4)
        self.fc = nn.Sequential(
            nn.Linear(channels, mid),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c = x.shape[:2]
        w = self.pool(x).view(b, c)
        w = self.fc(w).view(b, c, 1, 1)
        return x * w


# -----------------------------
# Spectral Embedding
# -----------------------------
class SpectralEmbedding(nn.Module):
    def __init__(self, in_channels=3, out_channels=64):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )
        self.se = SEBlock(out_channels)

    def forward(self, x):
        return self.se(self.proj(x))


# -----------------------------
# Residual Block
# -----------------------------
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        half = channels // 2

        self.b3 = nn.Sequential(
            nn.Conv2d(channels, half, 3, padding=1, bias=False),
            nn.BatchNorm2d(half),
            nn.GELU(),
            nn.Conv2d(half, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )

        self.b5 = nn.Sequential(
            nn.Conv2d(channels, half, 5, padding=2, bias=False),
            nn.BatchNorm2d(half),
            nn.GELU(),
            nn.Conv2d(half, channels, 5, padding=2, bias=False),
            nn.BatchNorm2d(channels),
        )

        self.act = nn.GELU()

    def forward(self, x):
        return self.act(x + self.b3(x) + self.b5(x))


# -----------------------------
# CNN Backbone
# -----------------------------
class CNNBackbone(nn.Module):
    def __init__(self, base_ch=64):
        super().__init__()

        def stage(cin, cout):
            return nn.Sequential(
                nn.Conv2d(cin, cout, 3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(cout),
                nn.GELU(),
                ResidualBlock(cout),
                SEBlock(cout),
            )

        self.s1 = stage(base_ch, base_ch * 2)
        self.s2 = stage(base_ch * 2, base_ch * 4)
        self.s3 = stage(base_ch * 4, base_ch * 8)

    def forward(self, x):
        c1 = self.s1(x)
        c2 = self.s2(c1)
        c3 = self.s3(c2)
        return c1, c2, c3


# -----------------------------
# FPN
# -----------------------------
class FPN(nn.Module):
    def __init__(self, base_ch=64, out_ch=256):
        super().__init__()

        self.lat3 = nn.Conv2d(base_ch * 8, out_ch, 1, bias=False)
        self.lat2 = nn.Conv2d(base_ch * 4, out_ch, 1, bias=False)
        self.lat1 = nn.Conv2d(base_ch * 2, out_ch, 1, bias=False)

        self.smooth = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.GELU()

    def _up_add(self, x, y):
        return F.interpolate(x, size=y.shape[-2:], mode='bilinear', align_corners=False) + y

    def forward(self, c1, c2, c3):
        p3 = self.lat3(c3)
        p2 = self._up_add(p3, self.lat2(c2))
        p1 = self._up_add(p2, self.lat1(c1))
        return self.act(self.bn(self.smooth(p1)))


# -----------------------------
# Transformer Block
# -----------------------------
class TransformerBlock(nn.Module):
    def __init__(self, dim, heads, mlp_ratio=4.0, dropout=0.1):
        super().__init__()

        self.n1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)

        self.n2 = nn.LayerNorm(dim)
        mid = int(dim * mlp_ratio)

        self.mlp = nn.Sequential(
            nn.Linear(dim, mid),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mid, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        h, _ = self.attn(self.n1(x), self.n1(x), self.n1(x))
        x = x + h
        x = x + self.mlp(self.n2(x))
        return x


# -----------------------------
# CNN-Transformer Fusion
# -----------------------------
class CNNTransformerFusion(nn.Module):
    def __init__(self, fpn_ch=256, dim=256, heads=8, n_layers=4, patch_size=8):
        super().__init__()

        self.ps = patch_size
        self.proj = nn.Linear(fpn_ch * patch_size * patch_size, dim)

        self.blocks = nn.ModuleList([
            TransformerBlock(dim, heads) for _ in range(n_layers)
        ])

        self.norm = nn.LayerNorm(dim)
        self.cnn_proj = nn.Conv2d(fpn_ch, dim, 1, bias=False)
        self.bn = nn.BatchNorm2d(dim)

    def forward(self, x):
        B, C, H, W = x.shape
        P = self.ps

        tokens = rearrange(x, 'b c (h p1) (w p2) -> b (h w) (c p1 p2)', p1=P, p2=P)
        tokens = self.proj(tokens)

        for blk in self.blocks:
            tokens = blk(tokens)

        tokens = self.norm(tokens)

        hp, wp = H // P, W // P
        D = tokens.shape[-1]

        t_map = tokens.view(B, hp, wp, D).permute(0, 3, 1, 2)
        t_up = F.interpolate(t_map, size=(H, W), mode='bilinear', align_corners=False)

        return self.bn(self.cnn_proj(x) + t_up)


# -----------------------------
# Cross Attention
# -----------------------------
class CrossAttentionRefinement(nn.Module):
    def __init__(self, dim, num_classes, heads=8):
        super().__init__()

        self.cls_tok = nn.Parameter(torch.randn(1, num_classes, dim))
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        B, C, H, W = x.shape

        tokens = x.view(B, C, -1).permute(0, 2, 1)
        cls = self.cls_tok.expand(B, -1, -1)

        refined, _ = self.attn(cls, tokens, tokens)
        return self.norm(refined + cls)


# -----------------------------
# FINAL MODEL
# -----------------------------
class AgroVision(nn.Module):
    def __init__(self, in_ch=3, num_classes=4):
        super().__init__()

        self.spectral = SpectralEmbedding(in_ch, 64)
        self.backbone = CNNBackbone(64)
        self.fpn = FPN(64, 256)
        self.fusion = CNNTransformerFusion(256, 256)
        self.cross_attn = CrossAttentionRefinement(256, num_classes)

        self.head = nn.Sequential(
            nn.LayerNorm(256),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.spectral(x)
        c1, c2, c3 = self.backbone(x)
        fpn_out = self.fpn(c1, c2, c3)
        fused = self.fusion(fpn_out)

        cls_emb = self.cross_attn(fused)
        logits = self.head(cls_emb)

        logits = logits.diagonal(dim1=1, dim2=2)
        confidence = torch.softmax(logits, -1).max(-1).values

        return logits, confidence