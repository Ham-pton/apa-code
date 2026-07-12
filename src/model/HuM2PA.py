# -*- coding: utf-8 -*-

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def masked_softmax(logits, mask, dim=-1):
    if mask is None:
        return torch.softmax(logits, dim=dim)

    mask = mask.bool().to(logits.device)
    logits = logits.masked_fill(~mask, torch.finfo(logits.dtype).min)
    weights = torch.softmax(logits, dim=dim)
    weights = weights * mask.to(weights.dtype)
    return weights / weights.sum(dim=dim, keepdim=True).clamp_min(1e-12)


class MultiAspectAttention(nn.Module):
    """Cross-aspect attention used at word and utterance levels."""

    def __init__(self, embed_dim, init_std=0.01):
        super().__init__()
        self.att_v = nn.Parameter(torch.randn(embed_dim) * init_std)
        self.att_W = nn.Parameter(torch.randn(embed_dim, embed_dim) * init_std)

    def forward(self, target, context, context_mask=None):
        if target.size(0) != context.size(0):
            raise ValueError("target and context must have the same batch size")
        if target.size(-1) != context.size(-1):
            raise ValueError("target and context must have the same embedding size")

        hidden = torch.tanh(torch.matmul(context, self.att_W))
        pool_logits = torch.tensordot(self.att_v, hidden, dims=([0], [2]))
        pool_mask = None if context_mask is None else context_mask.bool()
        pool_weight = masked_softmax(pool_logits, pool_mask, dim=1)
        weighted_context = context * pool_weight.unsqueeze(-1)

        score = torch.bmm(target, weighted_context.transpose(1, 2))
        pair_mask = None if pool_mask is None else pool_mask[:, None, :]
        attn = masked_softmax(score, pair_mask, dim=-1)
        return torch.bmm(attn, weighted_context)


class MultiHeadSelfAttention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
    ):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim={dim} must be divisible by num_heads={num_heads}")

        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = qk_scale or self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, mask=None):
        batch_size, seq_len, dim = x.shape

        qkv = self.qkv(x).reshape(
            batch_size, seq_len, 3, self.num_heads, self.head_dim
        )
        q, k, v = qkv.permute(2, 0, 3, 1, 4)

        score = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        key_mask = None
        if mask is not None:
            key_mask = mask.bool().to(x.device)[:, None, None, :]

        attn = masked_softmax(score, key_mask, dim=-1)
        attn = self.attn_drop(attn)

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).reshape(batch_size, seq_len, dim)
        out = self.proj_drop(self.proj(out))

        if mask is not None:
            out = out * mask.unsqueeze(-1).to(out.dtype)
        return out


class GatedLocalTemporalEncoder(nn.Module):
    """Local convolution and nonlinear projection with gated residual update."""

    def __init__(self, dim, d_conv=4, expand=2, dropout=0.1, use_gate=True):
        super().__init__()
        hidden_dim = int(dim * expand)
        kernel_size = max(3, int(d_conv) * 2 + 1)

        self.use_gate = bool(use_gate)
        self.norm = nn.LayerNorm(dim)
        self.local_conv = nn.Sequential(
            nn.Conv1d(
                dim,
                dim,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                groups=dim,
            ),
            nn.GELU(),
            nn.Conv1d(dim, dim, kernel_size=1),
        )
        self.value_proj = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
        )
        self.gate = nn.Sequential(nn.Linear(dim * 2, dim), nn.Sigmoid())
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        if mask is None:
            mask_value = torch.ones(
                x.size(0), x.size(1), 1, device=x.device, dtype=x.dtype
            )
        else:
            mask_value = mask.bool().to(x.device).unsqueeze(-1).to(x.dtype)

        residual = x * mask_value
        hidden = self.norm(residual) * mask_value
        local = self.local_conv(hidden.transpose(1, 2)).transpose(1, 2)
        delta = local + self.value_proj(hidden)

        if self.use_gate:
            delta = self.gate(torch.cat([residual, delta], dim=-1)) * delta

        return (residual + self.dropout(delta)) * mask_value


class IntraWordAttention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=4,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.1,
        proj_drop=0.0,
    ):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim={dim} must be divisible by num_heads={num_heads}")

        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = qk_scale or self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, attn_mask):
        batch_size, seq_len, dim = x.shape
        if attn_mask is None or attn_mask.shape != (batch_size, seq_len, seq_len):
            raise ValueError(
                f"attn_mask must have shape {(batch_size, seq_len, seq_len)}"
            )

        qkv = self.qkv(x).reshape(
            batch_size, seq_len, 3, self.num_heads, self.head_dim
        )
        q, k, v = qkv.permute(2, 0, 3, 1, 4)

        score = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        mask = attn_mask.bool().to(x.device)[:, None, :, :]
        attn = self.attn_drop(masked_softmax(score, mask, dim=-1))

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).reshape(batch_size, seq_len, dim)
        out = self.proj_drop(self.proj(out))

        valid_query = mask.squeeze(1).any(dim=-1).unsqueeze(-1)
        return out * valid_query.to(out.dtype)


class DynamicPhonemeRelation(nn.Module):
    """Adaptive phoneme graph with a same-word prior."""

    def __init__(
        self,
        dim,
        num_heads=1,
        dropout=0.1,
        same_word_bias=1.0,
        init_scale=0.1,
    ):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim={dim} must be divisible by num_heads={num_heads}")

        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.same_word_bias = float(same_word_bias)

        self.norm = nn.LayerNorm(dim)
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Sequential(nn.Linear(dim, dim), nn.Dropout(dropout))
        self.gate = nn.Sequential(nn.Linear(dim * 2, dim), nn.Sigmoid())
        self.scale = nn.Parameter(torch.tensor(float(init_scale)))

    def forward(self, x, word_pos, valid_mask):
        batch_size, seq_len, dim = x.shape
        valid_mask = valid_mask.bool().to(x.device)
        mask_value = valid_mask.unsqueeze(-1).to(x.dtype)

        hidden = self.norm(x)
        q = self.q_proj(hidden).view(
            batch_size, seq_len, self.num_heads, self.head_dim
        ).transpose(1, 2)
        k = self.k_proj(hidden).view(
            batch_size, seq_len, self.num_heads, self.head_dim
        ).transpose(1, 2)
        v = self.v_proj(hidden).view(
            batch_size, seq_len, self.num_heads, self.head_dim
        ).transpose(1, 2)

        score = torch.matmul(q, k.transpose(-2, -1))
        score = score / math.sqrt(float(self.head_dim))

        pair_mask = (
            valid_mask[:, None, :, None] & valid_mask[:, None, None, :]
        )

        if word_pos is not None:
            word_pos = word_pos.long().to(x.device)
            same_word = word_pos.unsqueeze(-1).eq(word_pos.unsqueeze(-2))
            same_word = same_word & valid_mask.unsqueeze(-1)
            same_word = same_word & valid_mask.unsqueeze(-2)
            score = score + same_word[:, None].to(score.dtype) * self.same_word_bias

        adjacency = torch.sigmoid(score) * pair_mask.to(score.dtype)
        identity = torch.eye(
            seq_len, device=x.device, dtype=torch.bool
        ).view(1, 1, seq_len, seq_len)
        adjacency = adjacency.masked_fill(identity & pair_mask, 1.0)
        adjacency = adjacency / adjacency.sum(dim=-1, keepdim=True).clamp_min(1e-6)

        message = torch.matmul(adjacency, v)
        message = message.transpose(1, 2).reshape(batch_size, seq_len, dim)
        message = self.out_proj(message)

        gate = self.gate(torch.cat([x, message], dim=-1))
        return (x + self.scale * gate * message) * mask_value


class ScoreGuidedAttentionPooling(nn.Module):
    def __init__(self, dim, score_dim=4, hidden_dim=None):
        super().__init__()
        hidden_dim = hidden_dim or dim
        self.score_norm = nn.LayerNorm(score_dim)
        self.attn = nn.Sequential(
            nn.Linear(dim + score_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x, score, mask):
        mask = mask.bool().to(x.device)
        score = torch.where(torch.isfinite(score), score, torch.zeros_like(score))
        score = self.score_norm(score.float()).to(x.dtype)

        logits = self.attn(torch.cat([x, score], dim=-1)).squeeze(-1)
        logits = logits / math.sqrt(float(x.size(-1)))
        weights = masked_softmax(logits, mask, dim=-1)
        return torch.bmm(weights.unsqueeze(1), x).squeeze(1)


class CanonicalPhoneGuidedFusion(nn.Module):
    """Canonical-phone-guided fusion of GOP-energy, HuPER and SSL features."""

    def __init__(
        self,
        embed_dim,
        gop_dim=84,
        energy_dim=7,
        huper_dim=5,
        ssl_dim=1024,
        use_energy=True,
        use_huper=True,
        use_ssl=True,
        dropout=0.1,
    ):
        super().__init__()
        self.gop_dim = int(gop_dim)
        self.energy_dim = int(energy_dim)
        self.huper_dim = int(huper_dim)
        self.ssl_dim = int(ssl_dim)
        self.use_energy = bool(use_energy)
        self.use_huper = bool(use_huper)
        self.use_ssl = bool(use_ssl)

        ge_dim = self.gop_dim + (self.energy_dim if self.use_energy else 0)
        self.expected_dim = ge_dim
        if self.use_huper:
            self.expected_dim += self.huper_dim
        if self.use_ssl:
            self.expected_dim += self.ssl_dim

        self.ge_proj = self._branch(ge_dim, embed_dim, dropout)
        self.huper_proj = (
            self._branch(self.huper_dim, embed_dim, dropout)
            if self.use_huper
            else None
        )
        self.ssl_proj = (
            self._branch(self.ssl_dim, embed_dim, dropout)
            if self.use_ssl
            else None
        )

        self.num_branches = 1 + int(self.use_huper) + int(self.use_ssl)
        gate_dim = embed_dim * (self.num_branches + 1)
        self.gate = nn.Sequential(
            nn.LayerNorm(gate_dim),
            nn.Linear(gate_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, self.num_branches),
        )
        self.out_norm = nn.LayerNorm(embed_dim)
        self.out_drop = nn.Dropout(dropout)
        self.last_gate = None

    @staticmethod
    def _branch(input_dim, embed_dim, dropout):
        return nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(self, features, phone_embedding, valid_mask=None):
        if features.size(-1) != self.expected_dim:
            raise ValueError(
                f"expected {self.expected_dim} input features, "
                f"but received {features.size(-1)}"
            )

        offset = 0
        gop = features[:, :, offset : offset + self.gop_dim]
        offset += self.gop_dim

        ge_parts = [gop]
        if self.use_energy:
            energy = features[:, :, offset : offset + self.energy_dim]
            offset += self.energy_dim
            ge_parts.append(energy)

        branches = [self.ge_proj(torch.cat(ge_parts, dim=-1))]

        if self.use_huper:
            huper = features[:, :, offset : offset + self.huper_dim]
            offset += self.huper_dim
            branches.append(self.huper_proj(huper))

        if self.use_ssl:
            ssl = features[:, :, offset : offset + self.ssl_dim]
            branches.append(self.ssl_proj(ssl))

        gate_input = torch.cat(branches + [phone_embedding], dim=-1)
        gate = torch.softmax(self.gate(gate_input), dim=-1)
        self.last_gate = gate.detach()

        fused = torch.zeros_like(branches[0])
        for index, branch in enumerate(branches):
            fused = fused + gate[:, :, index : index + 1] * branch

        fused = self.out_norm(self.out_drop(fused))
        if valid_mask is not None:
            fused = fused * valid_mask.unsqueeze(-1).to(fused.dtype)
        return fused


class HuPERAPA(nn.Module):
    """Multi-aspect, multi-granularity APA model with HuPER features."""

    def __init__(
        self,
        embed_dim,
        depth,
        input_dim=84,
        num_heads=4,
        dur_dim=1,
        use_dur=True,
        sequence_layers=2,
        sequence_d_conv=4,
        sequence_expand=2,
        sequence_gate=True,
        use_word_sequence=True,
        graph_max_words=16,
        use_adaptive_phone_graph=True,
        phone_graph_same_word_bias=1.0,
        phone_graph_scale=0.1,
        gop_dim=84,
        energy_dim=7,
        huper_dim=5,
        ssl_dim=1024,
        use_energy=True,
        use_huper=True,
        use_ssl=True,
        use_evidence_fusion=True,
        evidence_fusion_dropout=0.1,
    ):
        super().__init__()
        del graph_max_words

        self.input_dim = int(input_dim)
        self.embed_dim = int(embed_dim)
        self.dur_dim = int(dur_dim)
        self.use_dur = bool(use_dur)
        self.use_adaptive_phone_graph = bool(use_adaptive_phone_graph)
        self.use_evidence_fusion = bool(use_evidence_fusion)

        self.phone_proj = nn.Linear(40, embed_dim)
        self.input_proj = nn.Linear(input_dim, embed_dim)

        self.feature_fusion = (
            CanonicalPhoneGuidedFusion(
                embed_dim=embed_dim,
                gop_dim=gop_dim,
                energy_dim=energy_dim,
                huper_dim=huper_dim,
                ssl_dim=ssl_dim,
                use_energy=use_energy,
                use_huper=use_huper,
                use_ssl=use_ssl,
                dropout=evidence_fusion_dropout,
            )
            if self.use_evidence_fusion
            else None
        )

        self.duration_proj = (
            nn.Sequential(
                nn.Linear(self.dur_dim, embed_dim),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(embed_dim, embed_dim),
            )
            if self.use_dur
            else None
        )

        self.phone_lstm_norm = nn.LayerNorm(embed_dim)
        self.phone_lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=embed_dim,
            num_layers=depth,
            batch_first=True,
        )
        self.phone_dropout = nn.Dropout(0.1)

        self.phone_temporal = nn.ModuleList(
            [
                GatedLocalTemporalEncoder(
                    dim=embed_dim,
                    d_conv=sequence_d_conv,
                    expand=sequence_expand,
                    dropout=0.1,
                    use_gate=sequence_gate,
                )
                for _ in range(int(sequence_layers))
            ]
        )

        self.phone_conv_norm = nn.LayerNorm(embed_dim)
        self.phone_conv = nn.Conv1d(embed_dim, embed_dim, kernel_size=5, padding=2)

        self.phone_relation = (
            DynamicPhonemeRelation(
                dim=embed_dim,
                num_heads=num_heads,
                dropout=0.1,
                same_word_bias=phone_graph_same_word_bias,
                init_scale=phone_graph_scale,
            )
            if self.use_adaptive_phone_graph
            else None
        )

        self.phone_head = nn.Sequential(
            nn.LayerNorm(embed_dim * 2),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim, 1),
        )

        self.word_attn_norm = nn.LayerNorm(embed_dim)
        self.word_attn = IntraWordAttention(
            dim=embed_dim,
            num_heads=num_heads,
            attn_drop=0.1,
        )
        self.word_attn_drop = nn.Dropout(0.1)

        self.word_temporal = nn.ModuleList(
            [
                GatedLocalTemporalEncoder(
                    dim=embed_dim,
                    d_conv=sequence_d_conv,
                    expand=sequence_expand,
                    dropout=0.1,
                    use_gate=sequence_gate,
                )
            ]
            if use_word_sequence
            else []
        )

        self.word_fusion_gates = nn.ModuleList(
            [
                nn.Sequential(nn.Linear(embed_dim * 2, embed_dim), nn.Sigmoid())
                for _ in range(3)
            ]
        )
        self.word_rep_norms = nn.ModuleList(
            [nn.LayerNorm(embed_dim) for _ in range(3)]
        )
        self.word_rep_projs = nn.ModuleList(
            [nn.Linear(embed_dim, embed_dim) for _ in range(3)]
        )
        self.word_rep_drop = nn.Dropout(0.1)
        self.word_aspect_attn = MultiAspectAttention(embed_dim)
        self.word_aspect_drop = nn.Dropout(0.1)
        self.word_heads = nn.ModuleList(
            [
                nn.Sequential(nn.LayerNorm(embed_dim), nn.Linear(embed_dim, 1))
                for _ in range(3)
            ]
        )

        self.word_aspect_logits = nn.Parameter(torch.zeros(3))
        self.utt_temporal = nn.ModuleList(
            [
                GatedLocalTemporalEncoder(
                    dim=embed_dim,
                    d_conv=sequence_d_conv,
                    expand=sequence_expand,
                    dropout=0.1,
                    use_gate=sequence_gate,
                )
                for _ in range(max(1, int(sequence_layers) // 2))
            ]
        )
        self.utt_attn_norm = nn.LayerNorm(embed_dim)
        self.utt_attn = MultiHeadSelfAttention(
            dim=embed_dim,
            num_heads=num_heads,
            attn_drop=0.2,
        )
        self.utt_attn_drop = nn.Dropout(0.1)

        self.utt_rep_norm = nn.LayerNorm(embed_dim)
        self.utt_rep_projs = nn.ModuleList(
            [nn.Linear(embed_dim, embed_dim) for _ in range(5)]
        )
        self.utt_rep_drop = nn.Dropout(0.1)
        self.utt_aspect_attn = MultiAspectAttention(embed_dim)
        self.utt_aspect_drop = nn.Dropout(0.1)

        self.utt_pools = nn.ModuleList(
            [ScoreGuidedAttentionPooling(embed_dim, score_dim=4) for _ in range(5)]
        )
        self.utt_heads = nn.ModuleList(
            [
                nn.Sequential(nn.LayerNorm(embed_dim), nn.Linear(embed_dim, 1))
                for _ in range(5)
            ]
        )

        self.last_feature_gate = None
        self.last_evidence_gate = None

    @staticmethod
    def make_word_mask(word_pos, valid_mask):
        valid_mask = valid_mask.bool()
        if word_pos is None:
            return valid_mask.unsqueeze(1) & valid_mask.unsqueeze(2)

        word_pos = word_pos.long().to(valid_mask.device)
        same_word = word_pos.unsqueeze(2).eq(word_pos.unsqueeze(1))
        return (
            same_word
            & valid_mask.unsqueeze(2)
            & valid_mask.unsqueeze(1)
        )

    def _phone_embedding(self, phone, mask):
        phone_safe = phone.clamp_min(-1)
        one_hot = F.one_hot(phone_safe + 1, num_classes=40).float()
        return self.phone_proj(one_hot) * mask

    def _duration_embedding(self, duration, mask, batch_size, seq_len, device):
        if not self.use_dur:
            return torch.zeros(
                batch_size, seq_len, self.embed_dim, device=device, dtype=mask.dtype
            )

        if duration is None:
            return torch.zeros(
                batch_size, seq_len, self.embed_dim, device=device, dtype=mask.dtype
            )

        if duration.dim() == 2:
            duration = duration.unsqueeze(-1)
        if duration.shape[:2] != (batch_size, seq_len):
            raise ValueError("duration must have shape [B, T] or [B, T, D]")
        if duration.size(-1) != self.dur_dim:
            raise ValueError(
                f"duration dimension must be {self.dur_dim}, "
                f"but received {duration.size(-1)}"
            )

        duration = duration.float().to(device)
        duration = torch.where(
            torch.isfinite(duration), duration, torch.zeros_like(duration)
        )
        return self.duration_proj(duration * mask) * mask

    def forward(self, x, phn, word_pos=None, dur_feat=None):
        batch_size, seq_len, _ = x.shape
        phn = phn.long().to(x.device)
        if word_pos is not None:
            word_pos = word_pos.long().to(x.device)

        valid_mask = phn >= 0
        mask = valid_mask.unsqueeze(-1).to(x.dtype)

        phone_embedding = self._phone_embedding(phn, mask)

        if self.feature_fusion is not None:
            hidden = self.feature_fusion(x, phone_embedding, valid_mask)
            self.last_feature_gate = self.feature_fusion.last_gate
            self.last_evidence_gate = self.last_feature_gate
        else:
            hidden = self.input_proj(x) if x.size(-1) != self.embed_dim else x

        duration_embedding = self._duration_embedding(
            dur_feat, mask, batch_size, seq_len, x.device
        )
        hidden = (hidden + phone_embedding + duration_embedding) * mask

        self.phone_lstm.flatten_parameters()
        residual = hidden
        hidden = self.phone_lstm(self.phone_lstm_norm(hidden))[0]
        hidden = (residual + self.phone_dropout(hidden)) * mask

        for block in self.phone_temporal:
            hidden = block(hidden, valid_mask)

        residual = hidden
        conv = self.phone_conv(
            self.phone_conv_norm(hidden).transpose(1, 2)
        ).transpose(1, 2)
        hidden = (residual + self.phone_dropout(conv)) * mask

        if self.phone_relation is not None:
            hidden = self.phone_relation(hidden, word_pos, valid_mask)

        phone_score = self.phone_head(
            torch.cat([hidden, phone_embedding], dim=-1) * mask
        )

        word_mask = self.make_word_mask(word_pos, valid_mask)
        word_delta = self.word_attn(self.word_attn_norm(hidden), word_mask)
        word_context = (hidden + self.word_attn_drop(word_delta)) * mask

        for block in self.word_temporal:
            word_context = block(word_context, valid_mask)

        gate_input = torch.cat([hidden, word_context], dim=-1)
        word_representations = []
        for gate, norm, proj in zip(
            self.word_fusion_gates,
            self.word_rep_norms,
            self.word_rep_projs,
        ):
            weight = gate(gate_input)
            base = ((1.0 - weight) * hidden + weight * word_context) * mask
            representation = base + self.word_rep_drop(proj(norm(base)))
            word_representations.append(representation * mask)

        word_aspect_representations = []
        for index, target in enumerate(word_representations):
            other = torch.cat(
                word_representations[:index] + word_representations[index + 1 :],
                dim=1,
            )
            other_mask = torch.cat([valid_mask, valid_mask], dim=1)
            context = self.word_aspect_attn(target, other, other_mask)
            word_aspect_representations.append(
                (target + self.word_aspect_drop(context)) * mask
            )

        word_scores = [
            head(rep) for head, rep in zip(self.word_heads, word_aspect_representations)
        ]

        aspect_weight = torch.softmax(self.word_aspect_logits, dim=0)
        utterance_hidden = sum(
            aspect_weight[index] * word_aspect_representations[index]
            for index in range(3)
        )
        utterance_hidden = utterance_hidden * mask

        for block in self.utt_temporal:
            utterance_hidden = block(utterance_hidden, valid_mask)

        residual = utterance_hidden
        attn_out = self.utt_attn(
            self.utt_attn_norm(utterance_hidden), valid_mask
        )
        utterance_hidden = (
            residual + self.utt_attn_drop(attn_out)
        ) * mask

        shared = self.utt_rep_norm(utterance_hidden)
        utterance_representations = [
            (utterance_hidden + self.utt_rep_drop(proj(shared))) * mask
            for proj in self.utt_rep_projs
        ]

        utterance_aspect_representations = []
        for index, target in enumerate(utterance_representations):
            other = torch.cat(
                utterance_representations[:index]
                + utterance_representations[index + 1 :],
                dim=1,
            )
            other_mask = torch.cat([valid_mask] * 4, dim=1)
            context = self.utt_aspect_attn(target, other, other_mask)
            utterance_aspect_representations.append(
                (target + self.utt_aspect_drop(context)) * mask
            )

        guidance = torch.cat(
            [phone_score.detach()] + [score.detach() for score in word_scores],
            dim=-1,
        )
        guidance = guidance * mask

        utterance_scores = []
        for pool, head, rep in zip(
            self.utt_pools,
            self.utt_heads,
            utterance_aspect_representations,
        ):
            pooled = pool(rep, guidance, valid_mask)
            utterance_scores.append(head(pooled))

        return (
            utterance_scores[0],
            utterance_scores[1],
            utterance_scores[2],
            utterance_scores[3],
            utterance_scores[4],
            phone_score,
            word_scores[0],
            word_scores[1],
            word_scores[2],
        )

