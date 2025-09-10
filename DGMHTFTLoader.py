import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import warnings
from pytorch_forecasting import (
    TimeSeriesDataSet
)
from pytorch_forecasting import TemporalFusionTransformer
from pytorch_forecasting.metrics import QuantileLoss

class DualGateGatedResidualNetwork(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int,
                 dropout: float = 0.1, context_size: int = None):
        
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.context_size = context_size

        # Dual branches for X processing
        self.branch1_linear = nn.Linear(input_size, hidden_size)
        self.branch2_linear = nn.Linear(input_size, hidden_size)

        # Context integration if needed (static context cs)
        if context_size is not None:
            self.context_branch1 = nn.Linear(context_size, hidden_size, bias=False)
            self.context_branch2 = nn.Linear(context_size, hidden_size, bias=False)

        # Middle processing after concatenation
        self.middle_linear = nn.Linear(hidden_size*2, hidden_size)

        # Gate mechanism
        self.gate_linear = nn.Linear(hidden_size, output_size)

        # Skip connection and dropout
        self.skip_linear = nn.Linear(input_size, output_size)
        self.dropout = nn.Dropout(dropout)

        self.activation = nn.Tanh()

    def forward(self, x, context=None):
        # Residual/skip connection
        skip = self.skip_linear(x)

        # Branch 1: X -> Linear -> Tanh
        branch1 = self.activation(self.branch1_linear(x))

        # Branch 2: X -> Linear -> Tanh
        branch2 = self.activation(self.branch2_linear(x))

        # Context integration if provided (static context cs)
        if context is not None and self.context_size is not None:
            context1 = self.context_branch1(context)
            context2 = self.context_branch2(context)
            branch1 = branch1 + context1
            branch2 = branch2 + context2

        # Concatenate branches (paper mentions concatenation)
        combined = torch.cat([branch1, branch2], dim=-1)

        # Middle processing: Linear -> Tanh
        middle_out = self.activation(self.middle_linear(combined))

        # Gate mechanism
        gate = torch.sigmoid(self.gate_linear(middle_out))

        # Final output with residual connection
        output = skip + self.dropout(gate * middle_out)

        return output
    
class DualGateMultiHeadCrossAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        # Cross-attention components (Df as query, Ep as key/value)
        self.cross_q_proj = nn.Linear(d_model, d_model, bias=False)
        self.cross_k_proj = nn.Linear(d_model, d_model, bias=False)
        self.cross_v_proj = nn.Linear(d_model, d_model, bias=False)
        self.cross_out_proj = nn.Linear(d_model, d_model)

        # Self-attention components (on concatenated Cts)
        self.self_q_proj = nn.Linear(d_model, d_model, bias=False)
        self.self_k_proj = nn.Linear(d_model, d_model, bias=False)
        self.self_v_proj = nn.Linear(d_model, d_model, bias=False)
        self.self_out_proj = nn.Linear(d_model, d_model)

        # Final combination
        self.combine_proj = nn.Linear(d_model * 2, d_model)
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        # For storing attention weights
        self.last_cross_attn = None

    def scaled_dot_product_attention(self, q, k, v, mask=None):
        d_k = q.size(-1)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)

        if mask is not None:
            scores = scores.masked_fill(mask, -1e9)

        attn_weights = F.softmax(scores, dim=-1)
        output = torch.matmul(attn_weights, v)
        return output, attn_weights

    # def forward(self, encoder_output, decoder_output, mask=None, key_padding_mask=None):
    def forward(self, q, k, v, mask=None, key_padding_mask=None):

        encoder_output = k  # past features
        decoder_output = q  # future features

        batch_size, seq_len_dec, _ = decoder_output.shape
        seq_len_enc = encoder_output.shape[1]

        # ===== CROSS ATTENTION PATH =====
        # Paper: "only the output of future-known decoder serves as query,
        # and the output of past-observed encoder are transformed as key and value"

        # q = Df * Wq, k = Ep * Wk, v = Ep * Wv
        cross_q = self.cross_q_proj(decoder_output)  # [B, T_future, d_model]
        cross_k = self.cross_k_proj(encoder_output)  # [B, T_past, d_model]
        cross_v = self.cross_v_proj(encoder_output)  # [B, T_past, d_model]

        # Reshape for multi-head attention
        cross_q = cross_q.view(batch_size, seq_len_dec, self.n_heads, self.d_k).transpose(1, 2)
        cross_k = cross_k.view(batch_size, seq_len_enc, self.n_heads, self.d_k).transpose(1, 2)
        cross_v = cross_v.view(batch_size, seq_len_enc, self.n_heads, self.d_k).transpose(1, 2)

        # Apply cross attention with key padding mask
        cross_mask = None
        if key_padding_mask is not None:
            # Convert [B, T_past] to [B, 1, 1, T_past]
            cross_mask = key_padding_mask.unsqueeze(1).unsqueeze(2)

        cross_attn_out, cross_attn_weights = self.scaled_dot_product_attention(
            cross_q, cross_k, cross_v, mask=cross_mask
        )

        # Reshape back and apply output projection
        cross_attn_out = cross_attn_out.transpose(1, 2).contiguous().view(
            batch_size, seq_len_dec, self.d_model
        )
        cross_attn_out = self.cross_out_proj(cross_attn_out)

        # Store attention weights for interpretability
        self.last_cross_attn = cross_attn_weights.detach()

        # ===== SELF ATTENTION PATH =====
        # Paper: "the output of past-observed encoder is first concatenated
        # with the output of future-known decoder, and then the concatenated output Cts"

        # Create Cts by concatenation
        C_ts = torch.cat([encoder_output, decoder_output], dim=1)  # [B, T_past + T_future, d_model]

        # For self-attention, query comes from decoder part of Cts
        # "query is transformed from intercepted Cts related to the known-future time-series data"
        query_part = C_ts[:, seq_len_enc:, :]  # Extract decoder part

        self_q = self.self_q_proj(query_part)  # [B, T_future, d_model]
        self_k = self.self_k_proj(C_ts)        # [B, T_past + T_future, d_model]
        self_v = self.self_v_proj(C_ts)        # [B, T_past + T_future, d_model]

        # Reshape for multi-head attention
        seq_len_total = seq_len_enc + seq_len_dec
        self_q = self_q.view(batch_size, seq_len_dec, self.n_heads, self.d_k).transpose(1, 2)
        self_k = self_k.view(batch_size, seq_len_total, self.n_heads, self.d_k).transpose(1, 2)
        self_v = self_v.view(batch_size, seq_len_total, self.n_heads, self.d_k).transpose(1, 2)

        # Self-attention on concatenated sequence - no causal masking needed
        # This aligns with the paper's design for learning global temporal relationships
        self_mask = None

        self_attn_out, _ = self.scaled_dot_product_attention(
            self_q, self_k, self_v, mask=self_mask
        )

        # Reshape back and apply output projection
        self_attn_out = self_attn_out.transpose(1, 2).contiguous().view(
            batch_size, seq_len_dec, self.d_model
        )
        self_attn_out = self.self_out_proj(self_attn_out)

        # ===== COMBINE OUTPUTS =====
        # Paper: "the output of the self attention module is concatenated
        # with the output of the cross attention module"
        combined = torch.cat([cross_attn_out, self_attn_out], dim=-1)  # [B, T_future, 2*d_model]
        combined = self.combine_proj(combined)  # [B, T_future, d_model]

        # Residual connection and layer norm
        output = self.layer_norm(decoder_output + self.dropout(combined))

        return output, cross_attn_weights
    
class DGMHTFT(TemporalFusionTransformer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.save_hyperparameters(ignore=['loss', 'logging_metrics'])
        # Replace GRNs with DGRNs where paper specifies
        # Paper: "static context vectors are integrated with the outputs using DGRN"
        self.static_enrichment = DualGateGatedResidualNetwork(
            input_size=self.hparams.hidden_size,
            hidden_size=self.hparams.hidden_size,
            output_size=self.hparams.hidden_size,
            dropout=self.hparams.dropout,
            context_size=self.hparams.hidden_size  # For static context integration
        )

        # Position-wise feed-forward as DGRN (paper mentions this)
        self.pos_wise_ff = DualGateGatedResidualNetwork(
            input_size=self.hparams.hidden_size,
            hidden_size=self.hparams.hidden_size,
            output_size=self.hparams.hidden_size,
            dropout=self.hparams.dropout
        )

        # Replace multi-head attention with DGMCA
        self.multihead_attn = DualGateMultiHeadCrossAttention(
            d_model=self.hparams.hidden_size,
            n_heads=self.hparams.attention_head_size,
            dropout=self.hparams.dropout
        )

    def _decode(self, x: dict, static_context_decoder: torch.Tensor, context_decoder: torch.Tensor):
 
        # Get encoder outputs (past-observed encoder output Ep)
        encoder_output = x["encoder_output"]  # [B, T_past, hidden_size]

        # Future-known decoder output (Df)
        decoder_output = context_decoder  # [B, T_future, hidden_size]

        # Paper: "static context vectors are integrated with the outputs using DGRN"
        # Apply DGRN for static enrichment (integrating static context)
        enriched_decoder = self.static_enrichment(decoder_output, static_context_decoder)

        # Get attention masks from TFT framework
        decoder_mask = self.get_decoder_mask(enriched_decoder)
        key_padding_mask = self.get_attention_mask(encoder_output, x["encoder_lengths"])

        # Paper: "fed into the DGMCA for picking up long-range dependencies"
        attention_output, attention_weights = self.multihead_attn(
            q=enriched_decoder,  # future features as query
            k=encoder_output,    # past features as key
            v=encoder_output,    # past features as value
            mask=decoder_mask,
            key_padding_mask=key_padding_mask
        )

        # Apply gating mechanism from original TFT (maintain compatibility)
        if hasattr(self, 'post_attention_gate'):
            gated_output = self.post_attention_gate(attention_output, enriched_decoder)
        else:
            gated_output = attention_output

        # Paper: "non-linear processing in position-wise feed-forward layer"
        output = self.pos_wise_ff(gated_output)

        # Final output layer (unchanged from TFT)
        output = self.output_layer(output)

        # Handle attention weights format for TFT compatibility
        if attention_weights is None:
            batch_size = encoder_output.size(0)
            seq_len_enc = encoder_output.size(1)
            seq_len_dec = decoder_output.size(1)
            attention_weights = torch.zeros(
                batch_size, self.hparams.attention_head_size, seq_len_dec, seq_len_enc,
                device=encoder_output.device, dtype=encoder_output.dtype
            )

        return output, attention_weights