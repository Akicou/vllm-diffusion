# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Config for Nemotron-Labs-Sigma continuous diffusion language models."""

import math
from typing import Any

from transformers import PretrainedConfig

_DEFAULT_ROPE_PARAMETERS: dict[str, Any] = {
    "rope_type": "yarn",
    "type": "yarn",
    "rope_theta": 1000000.0,
    "factor": 16.0,
    "original_max_position_embeddings": 16384,
    "beta_fast": 32.0,
    "beta_slow": 1.0,
    "mscale": 1.0,
    "mscale_all_dim": 1.0,
    "llama_4_scaling_beta": 0.1,
}


class NemotronLabsSigmaConfig(PretrainedConfig):
    model_type = "nemotron_labs_sigma"
    is_continuous_diffusion = True

    def __init__(
        self,
        vocab_size: int = 131072,
        hidden_size: int = 4096,
        intermediate_size: int = 14336,
        num_hidden_layers: int = 34,
        num_attention_heads: int = 32,
        num_key_value_heads: int = 8,
        head_dim: int = 128,
        hidden_act: str = "silu",
        max_position_embeddings: int = 4096,
        rms_norm_eps: float = 1e-5,
        rope_theta: float = 1000000.0,
        rope_parameters: dict[str, Any] | None = None,
        embed_dim: int = 16,
        block_size: int = 32,
        replaid_selfcond: bool = True,
        replaid_clean_ar_embedding: bool = True,
        replaid_disable_output_prior: bool = False,
        replaid_disable_time_conditioning: bool = False,
        initializer_range: float = 0.02,
        tie_word_embeddings: bool = False,
        pad_token_id: int | None = None,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        schedule_parameterization: str = "legacy",
        schedule_degree: int = 32,
        native_family: str | None = None,
        native_nfe: int | None = None,
        include_ar_head: bool = False,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.hidden_act = hidden_act
        self.max_position_embeddings = max_position_embeddings
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        kwargs.pop("rope_scaling", None)
        rope = dict(rope_parameters or _DEFAULT_ROPE_PARAMETERS)
        rope.setdefault("rope_theta", rope_theta)
        self.rope_parameters = rope
        self.embed_dim = embed_dim
        self.block_size = block_size
        self.replaid_selfcond = replaid_selfcond
        self.replaid_clean_ar_embedding = replaid_clean_ar_embedding
        self.replaid_disable_output_prior = replaid_disable_output_prior
        self.replaid_disable_time_conditioning = replaid_disable_time_conditioning
        self.initializer_range = initializer_range
        self.schedule_parameterization = schedule_parameterization
        self.schedule_degree = schedule_degree
        self.native_family = native_family
        self.native_nfe = native_nfe
        self.include_ar_head = include_ar_head
        # vLLM detects diffusion models by this attribute.
        self.canvas_length = block_size
        super().__init__(
            tie_word_embeddings=tie_word_embeddings,
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            **kwargs,
        )

    @property
    def is_native_few_step(self) -> bool:
        return self.native_family == "phased_dmd"

    def backbone_rope_parameters(self) -> dict[str, Any]:
        """RoPE parameters in the form vLLM's YaRN embedding expects.

        The checkpoint's code derives the interpolation factor from
        `max_position_embeddings / original_max_position_embeddings` and the
        attention scale from the nominal `factor`, so the two are split here.
        """
        rope = self.rope_parameters
        original = rope["original_max_position_embeddings"]
        nominal = rope.get("factor", 1.0)

        def mscale(scale: float, m: float) -> float:
            return 0.1 * m * math.log(scale) + 1.0 if scale > 1 else 1.0

        attention_factor = mscale(nominal, rope.get("mscale", 1.0)) / mscale(
            nominal, rope.get("mscale_all_dim", 1.0)
        )
        return {
            "rope_type": "yarn",
            "rope_theta": rope["rope_theta"],
            "factor": self.max_position_embeddings / original,
            "original_max_position_embeddings": original,
            "beta_fast": rope["beta_fast"],
            "beta_slow": rope["beta_slow"],
            "attention_factor": attention_factor,
        }

    def backbone_llama_4_scaling(self) -> dict[str, Any] | None:
        rope = self.rope_parameters
        beta = rope.get("llama_4_scaling_beta")
        if beta is None:
            return None
        return {
            "original_max_position_embeddings": rope[
                "original_max_position_embeddings"
            ],
            "beta": beta,
        }
