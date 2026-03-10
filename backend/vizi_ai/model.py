"""
ViziGenesis vizi-o1 — Multi-Modal Market Transformer
=====================================================
Architecture overview
─────────────────────
                          ┌─────────────┐
                          │  Raw Input   │
                          └──────┬───────┘
               ┌─────────────────┼─────────────────────┐
               ▼                 ▼                     ▼
    ┌──────────────────┐ ┌──────────────┐ ┌────────────────────┐
    │  Price Encoder   │ │ Macro Encoder│ │  Context Encoders  │
    │  (Temporal Xfmr) │ │ (Temporal)   │ │ Fund + News + Mkt  │
    └────────┬─────────┘ └──────┬───────┘ └─────────┬──────────┘
             │                  │                    │
             ▼                  ▼                    ▼
    ┌───────────────────────────────────────────────────────────┐
    │    Cross-Modal Fusion Transformer  (with MoE option)     │
    │  (Learnable [CLS] token + cross-attention across all     │
    │   modality tokens, optional Mixture-of-Experts FFN)      │
    └───────────────────────────┬───────────────────────────────┘
                                │
               ┌────────────────┼────────────────┐
               ▼                ▼                ▼
         ┌──────────┐  ┌──────────────┐  ┌──────────┐
         │ Direction │  │ Return Heads │  │  Regime  │
         │  (↑/↓)   │  │ 1d, 5d, 21d  │  │ 3-class  │
         └──────────┘  └──────────────┘  └──────────┘

Design rationale
────────────────
1. **Modality-specific encoders** respect the different statistical
   properties of each data type.  Price series are non-stationary with
   fat-tailed returns; macro series are smooth and low-frequency;
   fundamentals are static snapshots; news is discrete text.

2. **Cross-modal fusion** via a shared transformer lets the model learn
   *how* macro shifts interact with price patterns and news sentiment —
   the core of "human-like" financial reasoning.

3. **Stock embeddings** allow the same weights to specialise per-stock
   while sharing cross-stock structure (transfer learning).

4. **Multi-task heads** provide richer gradient signal and capture the
   multi-faceted nature of market prediction: direction for trading,
   returns for sizing, regime for risk management.

5. **Gradient checkpointing** (optional) trades compute for VRAM,
   enabling ≥2× larger models on the same GPU.

6. **Mixture-of-Experts (MoE)** feed-forward layers (optional) scale
   capacity without proportional FLOPs increase: only top-k experts
   are active per token.

Scaling:
  • RTX 4090 (24 GB): d_model=256, n_layers=4   → ~25M params
  • H100 SXM (80 GB): d_model=512, n_layers=8   → ~120M params
  • H100 + MoE:       d_model=512, n_experts=8   → ~250M+ params (sparse)
"""
from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as activation_checkpoint


# ═══════════════════════════════════════════════════════════════
#  Config
# ═══════════════════════════════════════════════════════════════
class ModelConfig:
    """
    Central model hyperparameters.  Sized for RTX 4090 (24 GB) by default.

    Use ``ModelConfig.h100_preset()`` for H100 SXM (80 GB) scaling.
    Use ``ModelConfig.h100_moe_preset()`` for H100 + MoE (sparse 250M+).
    """
    # Core dimensions
    d_model: int = 256
    n_heads: int = 8
    n_layers: int = 4          # per-encoder depth
    fusion_layers: int = 4     # cross-modal fusion depth
    d_ff: int = 1024           # feed-forward inner dim
    dropout: float = 0.15

    # Per-modality input dims (set dynamically from data)
    n_price_features: int = 60  # overridden at runtime
    n_macro_features: int = 15
    n_market_features: int = 10
    n_fundamental_features: int = 20

    # Sequence lengths (must match DataConfig)
    price_seq_len: int = 120
    macro_seq_len: int = 60

    # News
    news_vocab_size: int = 8192
    news_embed_dim: int = 64
    max_news_items: int = 8
    max_news_tokens: int = 64

    # Stock universe
    n_stocks: int = 250

    # Targets
    n_return_horizons: int = 3   # 1d, 5d, 21d
    n_regimes: int = 3           # bear, sideways, bull

    # Gradient checkpointing (saves VRAM at cost of ~30% more compute)
    gradient_checkpointing: bool = False

    # Mixture-of-Experts
    use_moe: bool = False
    n_experts: int = 8
    top_k_experts: int = 2       # active experts per token
    moe_aux_loss_weight: float = 0.01  # load-balancing auxiliary loss

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            if hasattr(self, k):
                setattr(self, k, v)

    @classmethod
    def h100_preset(cls) -> "ModelConfig":
        """Dense H100 SXM (80 GB) preset: ~120M params."""
        return cls(
            d_model=512, n_heads=16, n_layers=8, fusion_layers=8,
            d_ff=2048, dropout=0.12, gradient_checkpointing=True,
        )

    @classmethod
    def h100_moe_preset(cls) -> "ModelConfig":
        """Sparse MoE H100 preset: ~250M+ total params, ~60M active per token."""
        return cls(
            d_model=512, n_heads=16, n_layers=8, fusion_layers=8,
            d_ff=2048, dropout=0.10, gradient_checkpointing=True,
            use_moe=True, n_experts=8, top_k_experts=2,
        )


# ═══════════════════════════════════════════════════════════════
#  Building blocks
# ═══════════════════════════════════════════════════════════════
class RotaryPositionalEncoding(nn.Module):
    """
    Rotary Position Embedding (RoPE) — better than sinusoidal for
    relative position awareness in financial time-series.
    """
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, d_model, 2).float() / d_model))
        t = torch.arange(max_len).float().unsqueeze(1)
        freqs = t * inv_freq.unsqueeze(0)          # (max_len, d/2)
        self.register_buffer("cos_cache", freqs.cos().unsqueeze(0))  # (1, max_len, d/2)
        self.register_buffer("sin_cache", freqs.sin().unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, D)"""
        T = x.size(1)
        d2 = x.size(2) // 2
        x1, x2 = x[..., :d2], x[..., d2:]
        cos = self.cos_cache[:, :T, :d2]
        sin = self.sin_cache[:, :T, :d2]
        return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)


class GatedResidual(nn.Module):
    """Gated residual block: gate selectively passes new information."""
    def __init__(self, d: int, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(d, d * 2)
        self.fc2 = nn.Linear(d, d)
        self.gate = nn.Linear(d, d)
        self.norm = nn.LayerNorm(d)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.gelu(self.fc1(x))
        h = h[..., :x.size(-1)]  # project back to d
        g = torch.sigmoid(self.gate(x))
        h = self.dropout(self.fc2(h))
        return self.norm(g * h + (1 - g) * x)


class TemporalBlock(nn.Module):
    """
    Single transformer encoder layer with RoPE.
    Pre-norm architecture (more stable for deep networks).
    Optionally uses MoE feed-forward.
    """
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float,
                 use_moe: bool = False, n_experts: int = 8, top_k: int = 2,
                 moe_aux_weight: float = 0.01):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)

        self.use_moe = use_moe
        if use_moe:
            self.ff = MoEFeedForward(d_model, d_ff, n_experts, top_k, dropout, moe_aux_weight)
        else:
            self.ff = nn.Sequential(
                nn.Linear(d_model, d_ff),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_ff, d_model),
                nn.Dropout(dropout),
            )

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        h = self.norm1(x)
        h, _ = self.attn(h, h, h, attn_mask=mask)
        x = x + h
        if self.use_moe:
            ff_out, _ = self.ff(self.norm2(x))
            x = x + ff_out
        else:
            x = x + self.ff(self.norm2(x))
        return x


# ═══════════════════════════════════════════════════════════════
#  Mixture-of-Experts Feed-Forward
# ═══════════════════════════════════════════════════════════════
class MoEFeedForward(nn.Module):
    """
    Top-k gated Mixture-of-Experts feed-forward layer.

    Each expert is a standard 2-layer FFN.  A lightweight router
    selects the top-k experts per token, enabling massive capacity
    without proportional FLOPs.  Includes an auxiliary load-balancing
    loss (Switch Transformer / GShard style).
    """
    def __init__(self, d_model: int, d_ff: int, n_experts: int = 8,
                 top_k: int = 2, dropout: float = 0.1,
                 aux_loss_weight: float = 0.01):
        super().__init__()
        self.n_experts = n_experts
        self.top_k = top_k
        self.aux_loss_weight = aux_loss_weight

        self.router = nn.Linear(d_model, n_experts, bias=False)
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_ff),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_ff, d_model),
                nn.Dropout(dropout),
            )
            for _ in range(n_experts)
        ])

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x: (B, T, D)
        Returns: (output, aux_loss)
        """
        B, T, D = x.shape
        x_flat = x.view(-1, D)  # (B*T, D)

        # Router logits → top-k selection
        logits = self.router(x_flat)                      # (B*T, n_experts)
        topk_vals, topk_idx = logits.topk(self.top_k, dim=-1)  # (B*T, top_k)
        topk_weights = F.softmax(topk_vals, dim=-1)       # (B*T, top_k)

        # Dispatch to experts
        output = torch.zeros_like(x_flat)
        for k in range(self.top_k):
            expert_idx = topk_idx[:, k]       # (B*T,)
            weight = topk_weights[:, k:k+1]   # (B*T, 1)
            for e in range(self.n_experts):
                mask = (expert_idx == e)
                if mask.any():
                    expert_input = x_flat[mask]
                    expert_output = self.experts[e](expert_input)
                    output[mask] += weight[mask] * expert_output

        # Load-balancing auxiliary loss (encourages uniform expert usage)
        router_probs = F.softmax(logits, dim=-1)          # (B*T, n_experts)
        avg_prob = router_probs.mean(dim=0)                # (n_experts,)
        # Fraction of tokens routed to each expert
        one_hot = F.one_hot(topk_idx[:, 0], self.n_experts).float()
        avg_frac = one_hot.mean(dim=0)                     # (n_experts,)
        aux_loss = self.aux_loss_weight * self.n_experts * (avg_prob * avg_frac).sum()

        return output.view(B, T, D), aux_loss


# ═══════════════════════════════════════════════════════════════
#  Modality encoders
# ═══════════════════════════════════════════════════════════════
class PriceEncoder(nn.Module):
    """
    Encodes OHLCV + technical indicator sequences.
    Supports gradient checkpointing for VRAM savings.
    """
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.gradient_checkpointing = cfg.gradient_checkpointing
        self.proj = nn.Linear(cfg.n_price_features, cfg.d_model)
        self.rope = RotaryPositionalEncoding(cfg.d_model, cfg.price_seq_len + 16)
        self.blocks = nn.ModuleList([
            TemporalBlock(cfg.d_model, cfg.n_heads, cfg.d_ff, cfg.dropout,
                          use_moe=cfg.use_moe, n_experts=cfg.n_experts,
                          top_k=cfg.top_k_experts, moe_aux_weight=cfg.moe_aux_loss_weight)
            for _ in range(cfg.n_layers)
        ])
        self.norm = nn.LayerNorm(cfg.d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.proj(x)
        h = self.rope(h)
        T = h.size(1)
        mask = torch.triu(torch.ones(T, T, device=h.device), diagonal=1).bool()
        for block in self.blocks:
            if self.gradient_checkpointing and self.training:
                h = activation_checkpoint(block, h, mask, use_reentrant=False)
            else:
                h = block(h, mask=mask)
        return self.norm(h)


class MacroEncoder(nn.Module):
    """Encodes macroeconomic indicator sequences."""
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.gradient_checkpointing = cfg.gradient_checkpointing
        self.proj = nn.Linear(cfg.n_macro_features, cfg.d_model)
        self.rope = RotaryPositionalEncoding(cfg.d_model, cfg.macro_seq_len + 16)
        n = max(cfg.n_layers // 2, 2)
        self.blocks = nn.ModuleList([
            TemporalBlock(cfg.d_model, cfg.n_heads, cfg.d_ff, cfg.dropout)
            for _ in range(n)
        ])
        self.norm = nn.LayerNorm(cfg.d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.proj(x)
        h = self.rope(h)
        for block in self.blocks:
            if self.gradient_checkpointing and self.training:
                h = activation_checkpoint(block, h, use_reentrant=False)
            else:
                h = block(h)
        return self.norm(h)


class MarketEncoder(nn.Module):
    """Encodes cross-market index / commodity / FX / crypto returns."""
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.gradient_checkpointing = cfg.gradient_checkpointing
        self.proj = nn.Linear(cfg.n_market_features, cfg.d_model)
        self.rope = RotaryPositionalEncoding(cfg.d_model, cfg.macro_seq_len + 16)
        self.blocks = nn.ModuleList([
            TemporalBlock(cfg.d_model, cfg.n_heads, cfg.d_ff, cfg.dropout)
            for _ in range(max(cfg.n_layers // 2, 2))
        ])
        self.norm = nn.LayerNorm(cfg.d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.proj(x)
        h = self.rope(h)
        for block in self.blocks:
            if self.gradient_checkpointing and self.training:
                h = activation_checkpoint(block, h, use_reentrant=False)
            else:
                h = block(h)
        return self.norm(h)


class FundamentalEncoder(nn.Module):
    """
    Encodes static fundamental features into d_model tokens.

    Fundamentals are point-in-time snapshots (not sequences), so we
    use a GatedResidual MLP to produce a single token that enters
    the fusion transformer.
    """
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cfg.n_fundamental_features, cfg.d_model),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            GatedResidual(cfg.d_model, cfg.dropout),
            GatedResidual(cfg.d_model, cfg.dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, n_fund) → (B, 1, d_model)"""
        return self.net(x).unsqueeze(1)


class NewsEncoder(nn.Module):
    """
    Encodes news headlines via learned token embeddings + attention pooling.

    Why hash-based tokens instead of a pretrained tokenizer?
    1. Zero external dependency (no HuggingFace tokenizer needed)
    2. The model learns task-specific embeddings from scratch
    3. Financial jargon is poorly served by general tokenizers
    4. Hash collisions are rare in VOCAB_SIZE=8192

    Each headline → bag of token embeddings → attention pool → one vector.
    Multiple headlines → sequence of vectors entering fusion.
    """
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.embed = nn.Embedding(cfg.news_vocab_size, cfg.news_embed_dim, padding_idx=0)
        self.headline_attn = nn.Linear(cfg.news_embed_dim, 1)
        self.proj = nn.Linear(cfg.news_embed_dim, cfg.d_model)
        self.norm = nn.LayerNorm(cfg.d_model)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        """
        ids: (B, max_news, max_tokens) int64
        Returns: (B, max_news, d_model)
        """
        B, N, T = ids.shape
        emb = self.embed(ids)  # (B, N, T, embed_dim)

        # Attention pool within each headline
        emb_flat = emb.view(B * N, T, -1)
        attn_logits = self.headline_attn(emb_flat)  # (B*N, T, 1)
        # Clamp logits to prevent exp() overflow in softmax — training
        # stability fix for sequences with many zero-padding tokens.
        attn_logits = attn_logits.clamp(-20.0, 20.0)
        attn_w = torch.softmax(attn_logits, dim=1)
        pooled = (emb_flat * attn_w).sum(dim=1)  # (B*N, embed_dim)
        pooled = pooled.view(B, N, -1)  # (B, N, embed_dim)

        # Project to d_model
        out = self.proj(pooled)
        return self.norm(out)  # (B, N, d_model)


# ═══════════════════════════════════════════════════════════════
#  Cross-Modal Fusion Transformer
# ═══════════════════════════════════════════════════════════════
class CrossModalFusion(nn.Module):
    """
    Concatenates tokens from all modality encoders and applies a
    shared transformer.  A learnable [CLS] token aggregates the
    final representation.  Supports gradient checkpointing and MoE.
    """
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.gradient_checkpointing = cfg.gradient_checkpointing
        self.cls_token = nn.Parameter(torch.randn(1, 1, cfg.d_model) * 0.02)

        # Modality-type embeddings (like token-type embeddings in BERT)
        # 0=CLS, 1=price, 2=macro, 3=market, 4=fundamental, 5=news, 6=stock
        self.modality_embed = nn.Embedding(7, cfg.d_model)

        self.blocks = nn.ModuleList([
            TemporalBlock(cfg.d_model, cfg.n_heads, cfg.d_ff, cfg.dropout,
                          use_moe=cfg.use_moe, n_experts=cfg.n_experts,
                          top_k=cfg.top_k_experts, moe_aux_weight=cfg.moe_aux_loss_weight)
            for _ in range(cfg.fusion_layers)
        ])
        self.norm = nn.LayerNorm(cfg.d_model)

    def forward(
        self,
        price_tokens: torch.Tensor,      # (B, T_price, D)
        macro_tokens: torch.Tensor,       # (B, T_macro, D)
        market_tokens: torch.Tensor,      # (B, T_macro, D)
        fund_token: torch.Tensor,         # (B, 1, D)
        news_tokens: torch.Tensor,        # (B, N_news, D)
        stock_token: torch.Tensor,        # (B, 1, D)
    ) -> torch.Tensor:
        """Returns the [CLS] representation: (B, D)"""
        B = price_tokens.size(0)
        device = price_tokens.device

        # Add modality-type embeddings
        cls = self.cls_token.expand(B, -1, -1) + self.modality_embed(torch.zeros(B, 1, dtype=torch.long, device=device))
        price_tokens = price_tokens + self.modality_embed(torch.ones(B, price_tokens.size(1), dtype=torch.long, device=device))
        macro_tokens = macro_tokens + self.modality_embed(torch.full((B, macro_tokens.size(1)), 2, dtype=torch.long, device=device))
        market_tokens = market_tokens + self.modality_embed(torch.full((B, market_tokens.size(1)), 3, dtype=torch.long, device=device))
        fund_token = fund_token + self.modality_embed(torch.full((B, 1), 4, dtype=torch.long, device=device))
        news_tokens = news_tokens + self.modality_embed(torch.full((B, news_tokens.size(1)), 5, dtype=torch.long, device=device))
        stock_token = stock_token + self.modality_embed(torch.full((B, 1), 6, dtype=torch.long, device=device))

        # Concatenate all tokens: [CLS, price..., macro..., market..., fund, news..., stock]
        seq = torch.cat([cls, price_tokens, macro_tokens, market_tokens, fund_token, news_tokens, stock_token], dim=1)

        for block in self.blocks:
            if self.gradient_checkpointing and self.training:
                seq = activation_checkpoint(block, seq, use_reentrant=False)
            else:
                seq = block(seq)

        # Extract CLS token
        cls_out = self.norm(seq[:, 0, :])  # (B, D)
        return cls_out


# ═══════════════════════════════════════════════════════════════
#  Output Heads
# ═══════════════════════════════════════════════════════════════
class PredictionHeads(nn.Module):
    """
    Multi-task prediction heads.

    Why multi-task?
    1. **Richer gradients**: more supervision signals = better representation
    2. **Regularisation**: auxiliary tasks prevent overfitting on direction
    3. **Practical**: a trader needs direction, magnitude, AND regime context

    Head outputs:
    - direction:  logit(price goes up)       (binary, BCEWithLogits)
    - ret_Xd:    expected X-day return       (regression, Huber)
    - regime:    bull/bear/sideways           (3-class, CE)
    - confidence: model's self-assessed certainty (auxiliary)
    """
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        d = cfg.d_model

        self.direction = nn.Sequential(
            nn.Linear(d, d // 2), nn.GELU(), nn.Dropout(cfg.dropout),
            nn.Linear(d // 2, d // 4), nn.GELU(),
            nn.Linear(d // 4, 1),
        )

        self.return_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d, d // 2), nn.GELU(), nn.Dropout(cfg.dropout * 0.5),
                nn.Linear(d // 2, d // 4), nn.GELU(),
                nn.Linear(d // 4, 1),
            )
            for _ in range(cfg.n_return_horizons)
        ])

        self.regime = nn.Sequential(
            nn.Linear(d, d // 2), nn.GELU(), nn.Dropout(cfg.dropout),
            nn.Linear(d // 2, cfg.n_regimes),
        )

        self.confidence = nn.Sequential(
            nn.Linear(d, d // 4), nn.GELU(),
            nn.Linear(d // 4, 1), nn.Sigmoid(),
        )

    def forward(self, cls: torch.Tensor) -> Dict[str, torch.Tensor]:
        return {
            "direction": self.direction(cls),
            "ret_1d": self.return_heads[0](cls),
            "ret_5d": self.return_heads[1](cls),
            "ret_21d": self.return_heads[2](cls),
            "regime": self.regime(cls),
            "confidence": self.confidence(cls),
        }


# ═══════════════════════════════════════════════════════════════
#  Full Model: ViziMarketTransformer
# ═══════════════════════════════════════════════════════════════
class ViziMarketTransformer(nn.Module):
    """
    The complete multi-modal, multi-stock market prediction model.

    Supports:
    - Dense mode: standard transformer FFN (~25-120M params)
    - Sparse MoE mode: Mixture-of-Experts FFN (~250M+ total, ~60M active)
    - Gradient checkpointing: saves ~50% VRAM for larger models
    - H100 presets: auto-configured for 80GB HBM3
    """

    def __init__(self, cfg: Optional[ModelConfig] = None):
        super().__init__()
        if cfg is None:
            cfg = ModelConfig()
        self.cfg = cfg

        # Stock embedding
        self.stock_embed = nn.Embedding(cfg.n_stocks, cfg.d_model)

        # Modality encoders
        self.price_enc = PriceEncoder(cfg)
        self.macro_enc = MacroEncoder(cfg)
        self.market_enc = MarketEncoder(cfg)
        self.fund_enc = FundamentalEncoder(cfg)
        self.news_enc = NewsEncoder(cfg)

        # Cross-modal fusion
        self.fusion = CrossModalFusion(cfg)

        # Prediction heads
        self.heads = PredictionHeads(cfg)

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # ── Input sanitization ──
        # Clamp all floating-point inputs to prevent NaN propagation.
        # Even with data-pipeline sanitization, edge cases (e.g. a corrupt
        # CSV row that slipped through) can produce extreme values that
        # cause NaN in the very first Linear projection.
        _CLAMP = 10.0
        price_seq = batch["price_seq"].clamp(-_CLAMP, _CLAMP)
        macro_seq = batch["macro_seq"].clamp(-_CLAMP, _CLAMP)
        market_seq = batch["market_seq"].clamp(-_CLAMP, _CLAMP)
        fundamental = batch["fundamental"].clamp(-_CLAMP, _CLAMP)

        # Encode each modality
        price_tokens = self.price_enc(price_seq)
        macro_tokens = self.macro_enc(macro_seq)
        market_tokens = self.market_enc(market_seq)
        fund_token = self.fund_enc(fundamental)
        news_tokens = self.news_enc(batch["news_ids"])
        stock_token = self.stock_embed(batch["stock_id"]).unsqueeze(1)

        # Fuse all modalities
        cls = self.fusion(
            price_tokens, macro_tokens, market_tokens,
            fund_token, news_tokens, stock_token,
        )

        # Predict
        return self.heads(cls)

    def count_parameters(self, only_trainable: bool = True) -> int:
        if only_trainable:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())

    def summary(self) -> str:
        total = self.count_parameters(only_trainable=False)
        trainable = self.count_parameters(only_trainable=True)
        parts = {
            "price_enc": sum(p.numel() for p in self.price_enc.parameters()),
            "macro_enc": sum(p.numel() for p in self.macro_enc.parameters()),
            "market_enc": sum(p.numel() for p in self.market_enc.parameters()),
            "fund_enc": sum(p.numel() for p in self.fund_enc.parameters()),
            "news_enc": sum(p.numel() for p in self.news_enc.parameters()),
            "fusion": sum(p.numel() for p in self.fusion.parameters()),
            "heads": sum(p.numel() for p in self.heads.parameters()),
            "stock_embed": sum(p.numel() for p in self.stock_embed.parameters()),
        }
        lines = [
            f"ViziMarketTransformer — {total:,} total params ({trainable:,} trainable)",
            f"  config: d_model={self.cfg.d_model}, n_layers={self.cfg.n_layers}, "
            f"fusion={self.cfg.fusion_layers}, d_ff={self.cfg.d_ff}",
        ]
        if self.cfg.use_moe:
            lines.append(
                f"  MoE: {self.cfg.n_experts} experts, top-{self.cfg.top_k_experts} active"
            )
        if self.cfg.gradient_checkpointing:
            lines.append("  gradient_checkpointing: enabled")
        for name, count in parts.items():
            lines.append(f"  {name:15s}: {count:>10,}  ({count/total*100:.1f}%)")
        return "\n".join(lines)
