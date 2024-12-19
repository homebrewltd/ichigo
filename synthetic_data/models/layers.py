from typing import Optional

from torch import Tensor, nn

from models.modules import LayerNorm, MultiHeadAttention


class ResidualAttentionBlock(nn.Module):
    """
    Residual Attention Block combining self-attention, optional cross-attention, and feed-forward layers.

    Args:
        n_state (int): Hidden state dimension
        n_head (int): Number of attention heads
        cross_attention (bool, optional): Whether to include cross-attention. Defaults to False.
        rope (bool, optional): Whether to use RoPE. Defaults to False.
        qk_scale (float, optional): Query-Key scaling factor. Defaults to 1.
        ffn_mult (int, optional): Multiplier for FFN intermediate dimension. Defaults to 4.
    """

    def __init__(
        self,
        n_state: int,
        n_head: int,
        cross_attention: bool = False,
        rope: bool = False,
        qk_scale: float = 1,
        ffn_mult: int = 4,
    ):
        super().__init__()
        self.attn = MultiHeadAttention(n_state, n_head, qk_scale=qk_scale, rope=rope)
        self.attn_ln = LayerNorm(n_state)

        self.cross_attn = (
            MultiHeadAttention(
                n_state, n_head, qk_scale=qk_scale, rope=rope, cross=True
            )
            if cross_attention
            else None
        )
        self.cross_attn_ln = LayerNorm(n_state) if cross_attention else None

        n_mlp = n_state * ffn_mult
        self.mlp = nn.Sequential(
            nn.Linear(n_state, n_mlp), nn.GELU(), nn.Linear(n_mlp, n_state)
        )
        self.mlp_ln = LayerNorm(n_state)

    def setup_kv_cache(self, max_batch_size, max_seq_len, max_cross_seq_len=None):
        self.attn.setup_kv_cache(max_batch_size, max_seq_len)
        if self.cross_attn:
            self.cross_attn.setup_kv_cache(max_batch_size, max_cross_seq_len)

    def forward(
        self,
        x: Tensor,
        x_positions: Tensor = None,
        xa: Optional[Tensor] = None,
        xa_positions: Optional[Tensor] = None,
        causal=False,
        mask=None,
    ):
        lnx = self.attn_ln(x)
        x = x + self.attn(lnx, x_positions, lnx, x_positions, causal=causal, mask=mask)
        if self.cross_attn:
            lnx = self.cross_attn_ln(x)
            x = x + self.cross_attn(lnx, x_positions, xa, xa_positions)
        x = x + self.mlp(self.mlp_ln(x))
        return x
