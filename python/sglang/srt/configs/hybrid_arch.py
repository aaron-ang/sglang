from __future__ import annotations

from sglang.srt.configs import (
    BailingHybridConfig,
    FalconH1Config,
    GraniteMoeHybridConfig,
    JetNemotronConfig,
    JetVLMConfig,
    KimiLinearConfig,
    Lfm2Config,
    Lfm2MoeConfig,
    Lfm2VlConfig,
    NemotronH_Nano_VL_V2_Config,
    NemotronHConfig,
    Qwen3_5Config,
    Qwen3_5MoeConfig,
    Qwen3NextConfig,
)


def qwen3_next_config(*, model_runner_ref):
    config = model_runner_ref.model_config.hf_config
    if isinstance(config, Qwen3NextConfig):
        return config
    return None


def hybrid_lightning_config(*, model_runner_ref):
    config = model_runner_ref.model_config.hf_config
    if isinstance(config, BailingHybridConfig):
        return config
    return None


def hybrid_gdn_config(*, model_runner_ref):
    config = model_runner_ref.model_config.hf_config.get_text_config()
    if isinstance(
        config,
        Qwen3NextConfig
        | Qwen3_5Config
        | Qwen3_5MoeConfig
        | JetNemotronConfig
        | JetVLMConfig,
    ):
        return config
    return None


def mamba2_config(*, model_runner_ref):
    config = model_runner_ref.model_config.hf_config
    if isinstance(config, NemotronHConfig) and model_runner_ref.is_draft_worker:
        # NemotronH MTP draft models have no Mamba layers (pattern like "*E")
        # so they shouldn't use HybridLinearAttnBackend
        pattern = getattr(config, "mtp_hybrid_override_pattern", None)
        if pattern is not None and "M" not in pattern:
            return None
    if isinstance(
        config,
        FalconH1Config | NemotronHConfig | Lfm2Config | Lfm2MoeConfig | Lfm2VlConfig,
    ):
        return config
    if isinstance(config, NemotronH_Nano_VL_V2_Config):
        return config.llm_config

    if isinstance(config, GraniteMoeHybridConfig):
        has_mamba = any(
            layer_type == "mamba" for layer_type in getattr(config, "layer_types", [])
        )
        if not has_mamba:
            return None
        else:
            return config

    return None


def kimi_linear_config(*, model_runner_ref):
    config = model_runner_ref.model_config.hf_config
    if isinstance(config, KimiLinearConfig):
        return config
    return None


def linear_attn_model_spec(*, model_runner_ref):
    result = model_runner_ref._get_linear_attn_registry_result()
    return result[0] if result else None


def mambaish_config(*, model_runner_ref):
    existing = (
        mamba2_config(model_runner_ref=model_runner_ref)
        or hybrid_gdn_config(model_runner_ref=model_runner_ref)
        or kimi_linear_config(model_runner_ref=model_runner_ref)
        or hybrid_lightning_config(model_runner_ref=model_runner_ref)
    )
    if existing:
        return existing
    result = model_runner_ref._get_linear_attn_registry_result()
    return result[1] if result else None
