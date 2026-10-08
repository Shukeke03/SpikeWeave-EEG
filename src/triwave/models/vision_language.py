"""C branch: shared medical text/image tokens and ViLT-style joint interaction.

Offline frontends are explicit untrained engineering fixtures. BiomedCLIP loads
an official local checkpoint strictly; it never silently substitutes random
weights. Numerical CWT is an EEG view, not an independent diagnostic image.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from triwave.config import ModelConfig
from triwave.signal import strict_patch_valid
from triwave.types import BranchOutput

from .attention import TransformerBlock, sinusoidal_embedding
from .encoders import check_geometry


class SharedVisionLanguageEncoder(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        d = config.condition_dim
        self.is_medically_pretrained = (
            config.frontend_backend == "biomedclip" and not config.frontend_test_fixture
        )
        self.medical = None
        if config.frontend_backend == "biomedclip":
            self._load_medical()
        else:
            self.image_stem = nn.Conv2d(1, d, config.visual_patch, config.visual_patch)
            self.visual_blocks = nn.ModuleList(
                TransformerBlock(d, config.visual_heads, 0.0) for _ in range(config.visual_depth)
            )
            self.text_embedding = nn.Embedding(config.text_vocab_size, d)
            self.text_blocks = nn.ModuleList(
                TransformerBlock(d, config.condition_heads, config.representation_dropout)
                for _ in range(config.text_depth)
            )
        self.numeric_reconstruction_head = nn.Linear(d, config.visual_patch**2)
        self.geometry = nn.Linear(6, d)
        self.coordinates = nn.Linear(2, d)
        self.coverage = nn.Linear(1, d)
        self.type_embedding = nn.Embedding(4, d)
        self.attributes = nn.Linear(1, d)
        self.attribute_index = nn.Embedding(max(config.attribute_dim, 1), d)
        self.align_token = nn.Parameter(torch.randn(1, 1, d) * 0.02)
        self.structure_token = nn.Parameter(torch.randn(1, 1, d) * 0.02)
        self.null_text = nn.Parameter(torch.randn(1, 1, d) * 0.02)
        self.null_image = nn.Parameter(torch.randn(1, 1, d) * 0.02)
        self.presence = nn.Linear(3, d)
        self.joint_blocks = nn.ModuleList(
            TransformerBlock(d, config.condition_heads, config.representation_dropout)
            for _ in range(config.fusion_depth)
        )
        self.norm = nn.LayerNorm(d)

        def head():
            return nn.Sequential(
                nn.Linear(d, config.projection_hidden_dim),
                nn.GELU(),
                nn.Dropout(config.representation_dropout),
                nn.Linear(config.projection_hidden_dim, config.projection_dim),
            )

        self.semantic_head, self.structural_head = head(), head()

    def _load_medical(self) -> None:
        cfg = self.config
        if not cfg.biomedclip_local_path:
            raise ValueError(
                "biomedclip requires biomedclip_local_path; run prepare-biomedclip first"
            )
        root = Path(cfg.biomedclip_local_path)
        configuration = root / "open_clip_config.json"
        weight = next(
            (
                root / name
                for name in ("open_clip_pytorch_model.bin", "model.safetensors")
                if (root / name).is_file()
            ),
            None,
        )
        text_root = Path(cfg.frontend_text_config_path or root / "text_config")
        if (
            not configuration.is_file()
            or weight is None
            or not (text_root / "config.json").is_file()
        ):
            raise FileNotFoundError(
                "BiomedCLIP needs official config/weights and local text_config/config.json"
            )
        import open_clip
        from open_clip.model import CustomTextCLIP

        content = json.loads(configuration.read_text())["model_cfg"]
        if not cfg.frontend_test_fixture and (
            content.get("embed_dim") != 512
            or content["vision_cfg"].get("timm_model_name") != "vit_base_patch16_224"
            or content["text_cfg"].get("hf_model_name")
            != "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract"
            or content["text_cfg"].get("context_length") != 256
            or cfg.condition_dim != 768
        ):
            raise ValueError(
                "production frontend requires the official BiomedCLIP architecture/config"
            )
        content["text_cfg"]["hf_model_name"] = str(text_root)
        content["text_cfg"]["hf_model_pretrained"] = False
        content["vision_cfg"]["timm_model_pretrained"] = False
        self.medical = CustomTextCLIP(**content)
        open_clip.load_checkpoint(self.medical, str(weight), strict=True)
        trunk = self.medical.visual.trunk
        if trunk.embed_dim != cfg.condition_dim or cfg.visual_patch != 16:
            raise ValueError("official BiomedCLIP requires condition_dim=768 and visual_patch=16")
        for parameter in self.medical.parameters():
            parameter.requires_grad_(False)
        old = trunk.patch_embed.proj
        self.image_stem = nn.Conv2d(
            1, old.out_channels, old.kernel_size, old.stride, bias=old.bias is not None
        )
        with torch.no_grad():
            self.image_stem.weight.copy_(old.weight.sum(1, keepdim=True))
            if old.bias is not None:
                self.image_stem.bias.copy_(old.bias)
        self.image_position = nn.Parameter(trunk.pos_embed.detach().clone())
        self.text_adapter = nn.Linear(self.medical.text.config.hidden_size, cfg.condition_dim)

    def train(self, mode: bool = True):
        super().train(mode)
        if self.medical is not None:
            self.medical.eval()
        return self

    @staticmethod
    def _medical_visual_block(block: nn.Module, tokens: Tensor, mask: Tensor) -> Tensor:
        """Apply an official ViT block with explicit EEG-view token validity."""
        value = torch.where(mask[..., None], block.norm1(tokens), 0)
        batch, count, dim = value.shape
        heads = block.attn.num_heads
        q, k, v = block.attn.qkv(value).reshape(batch, count, 3, heads, dim // heads).permute(
            2, 0, 3, 1, 4
        )
        q, k = block.attn.q_norm(q), block.attn.k_norm(k)
        attended = F.scaled_dot_product_attention(
            q, k, v, attn_mask=mask[:, None, None, :], dropout_p=0.0
        ).transpose(1, 2).reshape(batch, count, dim)
        tokens = tokens + block.ls1(block.attn.proj(attended))
        tokens = tokens + block.ls2(block.mlp(block.norm2(tokens)))
        return torch.where(mask[..., None], tokens, 0)

    def _visual(self, image: Tensor, valid: Tensor) -> tuple[Tensor, Tensor]:
        cfg = self.config
        b, c, f, t = image.shape
        if (c, f, t) != (cfg.channels, cfg.cwt_frequencies, cfg.cwt_times):
            raise ValueError("numerical CWT must match configured [B,C,F,T]")
        safe = torch.where(valid, image, 0)
        if not torch.isfinite(safe).all():
            raise ValueError("observed CWT must be finite")
        p = cfg.visual_patch
        token_valid = strict_patch_valid(valid, p).reshape(b * c, -1)
        # Partial patches are absent tokens, not partially observed summaries.
        coverage = token_valid.to(image.dtype)
        tokens = self.image_stem(safe.reshape(b * c, 1, f, t)).flatten(2).transpose(1, 2)
        tokens = torch.where(token_valid[..., None], tokens, 0)
        if self.medical is None:
            position = sinusoidal_embedding(
                torch.arange(tokens.shape[1], device=tokens.device, dtype=tokens.dtype),
                cfg.condition_dim,
            )
            tokens = tokens + position[None]
            for block in self.visual_blocks:
                tokens = block(tokens, token_valid)
        else:
            trunk = self.medical.visual.trunk
            side = math.isqrt(self.image_position.shape[1] - 1)
            pos = (
                self.image_position[:, 1:]
                .reshape(1, side, side, cfg.condition_dim)
                .permute(0, 3, 1, 2)
            )
            pos = F.interpolate(pos, (f // p, t // p), mode="bicubic", align_corners=False)
            pos = torch.cat((self.image_position[:, :1], pos.flatten(2).transpose(1, 2)), 1)
            cls = trunk.cls_token.expand(b * c, -1, -1)
            tokens = torch.cat((cls, tokens), 1) + pos.to(tokens.dtype)
            mask = torch.cat(
                (torch.ones(b * c, 1, device=valid.device, dtype=torch.bool), token_valid), 1
            )
            tokens = trunk.norm_pre(tokens)
            # Reuse pretrained ViT weights with explicit key/query missingness.
            for block in trunk.blocks:
                if cfg.activation_checkpointing and self.training and torch.is_grad_enabled():
                    tokens = checkpoint(self._medical_visual_block, block, tokens, mask,
                                        use_reentrant=False, preserve_rng_state=True)
                else:
                    tokens = self._medical_visual_block(block, tokens, mask)
            tokens = trunk.norm(tokens)[:, 1:]
        tokens = torch.where(token_valid[..., None], tokens, 0)
        return tokens.reshape(b, c, -1, cfg.condition_dim), coverage.reshape(b, c, -1)

    def numeric_reconstruction(self, image: Tensor, valid: Tensor, patch_mask: Tensor) -> Tensor:
        """Stage A numerical-view adaptation; hidden pixels never enter the stem."""
        b, c, f, t = image.shape
        p = self.config.visual_patch
        if patch_mask.shape != (b, c, f // p, t // p):
            raise ValueError("numeric reconstruction mask must match the image patch grid")
        pixel_mask = patch_mask.repeat_interleave(p, -2).repeat_interleave(p, -1)
        corrupted = torch.where(pixel_mask, 0, image)
        tokens, _ = self._visual(corrupted, valid)
        reconstructed = self.numeric_reconstruction_head(tokens)
        return (
            reconstructed.reshape(b, c, f // p, t // p, p, p)
            .permute(0, 1, 2, 4, 3, 5)
            .reshape_as(image)
        )

    def _text(self, ids: Tensor, valid: Tensor) -> Tensor:
        safe = torch.where(valid, ids, 0)
        if self.medical is not None:
            output = self.medical.text.transformer(input_ids=safe, attention_mask=valid.long())
            tokens = self.text_adapter(output.last_hidden_state)
        else:
            if ((safe < 0) | (safe >= self.config.text_vocab_size)).any():
                raise ValueError("observed token IDs outside vocabulary")
            tokens = self.text_embedding(safe)
            positions = torch.arange(ids.shape[1], device=ids.device, dtype=tokens.dtype)
            tokens = tokens + sinusoidal_embedding(positions, self.config.condition_dim)[None]
            for block in self.text_blocks:
                tokens = block(tokens, valid)
        return torch.where(valid[..., None], tokens, 0)

    def forward(
        self,
        structure: Tensor | None = None,
        structure_valid: Tensor | None = None,
        input_ids: Tensor | None = None,
        text_valid: Tensor | None = None,
        *,
        lead_geometry: Tensor,
        attribute_values: Tensor | None = None,
        attribute_valid: Tensor | None = None,
    ) -> BranchOutput:
        cfg = self.config
        b = len(lead_geometry)
        check_geometry(lead_geometry, b, cfg.channels)
        device = lead_geometry.device
        if structure is not None:
            if structure_valid is not None and structure_valid.shape != structure.shape:
                raise ValueError("CWT mask must match numeric image dimensions")
            iv = (
                torch.ones_like(structure, dtype=torch.bool)
                if structure_valid is None
                else structure_valid.bool()
            )
            image, coverage = self._visual(structure, iv)
            image_presence = coverage.flatten(1).bool().any(-1)
            rows, cols = cfg.cwt_frequencies // cfg.visual_patch, cfg.cwt_times // cfg.visual_patch
            hz = (
                torch.logspace(
                    math.log10(cfg.cwt_min_frequency),
                    math.log10(cfg.cwt_max_frequency),
                    cfg.cwt_frequencies,
                    device=device,
                )
                .reshape(rows, cfg.visual_patch)
                .mean(-1)
            )
            seconds = (
                (torch.arange(cols, device=device) + 0.5) * cfg.samples / cfg.sample_rate / cols
            )
            freq, time = torch.meshgrid(hz.log(), seconds, indexing="ij")
            coords = torch.stack((freq, time), -1).reshape(-1, 2).to(image.dtype)
            image = (
                image
                + self.geometry(lead_geometry.to(image.dtype))[:, :, None]
                + self.coordinates(coords)[None, None]
                + self.coverage(coverage[..., None].to(image.dtype))
            )
            values = image.reshape(b * cfg.channels, -1, cfg.condition_dim).transpose(1, 2)
            weights = coverage.reshape(b * cfg.channels, 1, -1).to(image.dtype)
            num = F.adaptive_avg_pool1d(values * weights, cfg.image_summary_tokens)
            den = F.adaptive_avg_pool1d(weights, cfg.image_summary_tokens)
            image = (num / den.clamp_min(1e-8)).transpose(1, 2).reshape(b, -1, cfg.condition_dim)
            imask = (den[:, 0] > 0).reshape(b, -1)
        else:
            image = self.null_image.expand(b, -1, -1)
            imask = torch.zeros(b, 1, dtype=torch.bool, device=device)
            image_presence = imask.any(-1)
        # Visible null modality slots for missing rows; presence is separate.
        image = image.clone()
        imask = imask.clone()
        image[:, 0] = torch.where(image_presence[:, None], image[:, 0], self.null_image[0, 0])
        imask[:, 0] |= ~image_presence
        if input_ids is not None:
            if (
                input_ids.ndim != 2
                or input_ids.shape[0] != b
                or input_ids.shape[1] > cfg.text_max_length
            ):
                raise ValueError("text tokens must match batch and configured context length")
            if text_valid is not None and text_valid.shape != input_ids.shape:
                raise ValueError("text mask must match tokens")
            tv = (input_ids != 0) if text_valid is None else text_valid.bool()
            text_presence = tv.any(-1)
            text = self._text(input_ids, tv)
            tmask = tv.clone()
            text[:, 0] = torch.where(text_presence[:, None], text[:, 0], self.null_text[0, 0])
            tmask[:, 0] |= ~text_presence
        else:
            text = self.null_text.expand(b, -1, -1)
            tmask = torch.ones(b, 1, dtype=torch.bool, device=device)
            text_presence = torch.zeros(b, dtype=torch.bool, device=device)
        if attribute_values is not None:
            if attribute_values.shape != (b, cfg.attribute_dim):
                raise ValueError("attribute values must match the configured schema width")
            if attribute_valid is not None and attribute_valid.shape != attribute_values.shape:
                raise ValueError("attribute mask must match values")
            av = (
                torch.zeros_like(attribute_values, dtype=torch.bool)
                if attribute_valid is None
                else attribute_valid.bool()
            )
            values = torch.where(av, attribute_values, 0)
            if not torch.isfinite(values).all():
                raise ValueError("observed attributes must be finite")
            for index, name in enumerate(cfg.attribute_schema[: cfg.attribute_dim]):
                observed = av[:, index]
                value = values[:, index]
                if (
                    name.endswith("_fraction")
                    and ((value[observed] < 0) | (value[observed] > 1)).any()
                ):
                    raise ValueError("power fractions must lie in [0,1]")
                if (
                    name.endswith("_hz")
                    and ((value[observed] < 0) | (value[observed] > cfg.sample_rate / 2)).any()
                ):
                    raise ValueError(
                        "frequency attributes must lie within the measurement Nyquist limit"
                    )
                if (
                    name in {"rms_uV", "std_uV", "peak_abs_uV", "line_length_uV"}
                    and (value[observed] < 0).any()
                ):
                    raise ValueError("amplitude magnitude attributes must be nonnegative")
            attrs = (
                self.attributes(values[..., None])
                + self.attribute_index.weight[None, : cfg.attribute_dim]
            )
            ap = av.any(-1)
        else:
            attrs = self.align_token.new_zeros(b, 1, cfg.condition_dim)
            av = torch.zeros(b, 1, dtype=torch.bool, device=device)
            ap = av.any(-1)
        presence = torch.stack((text_presence, image_presence, ap), -1)
        special = torch.cat((self.align_token, self.structure_token), 1).expand(b, -1, -1)
        special = special + self.presence(presence.to(special.dtype))[:, None]
        tokens = torch.cat(
            (
                special,
                text + self.type_embedding.weight[0],
                image + self.type_embedding.weight[1],
                attrs + self.type_embedding.weight[2],
            ),
            1,
        )
        mask = torch.cat((torch.ones(b, 2, dtype=torch.bool, device=device), tmask, imask, av), 1)
        for block in self.joint_blocks:
            tokens = block(tokens, mask)
        tokens = torch.where(mask[..., None], self.norm(tokens), 0)
        semantic = F.normalize(self.semantic_head(tokens[:, 0]), dim=-1, eps=1e-8)
        structural = F.normalize(self.structural_head(tokens[:, 1]), dim=-1, eps=1e-8)
        return BranchOutput(
            tokens=tokens,
            valid=mask,
            pooled=semantic,
            semantic=semantic,
            structural=structural,
            present=presence.any(-1),
            semantic_valid=text_presence,
            structural_valid=image_presence,
        )
