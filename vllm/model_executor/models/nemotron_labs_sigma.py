# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Nemotron-Labs-Sigma: blockwise continuous diffusion over token embeddings.

One Ministral-style backbone run in two modes, like DiffusionGemma:
- clean mode: causal attention over token embeddings, writes KV
- denoise mode: bidirectional attention over a block of noisy latents

A block starts as Gaussian noise in a low-dimensional embedding space. Each
engine step is one denoiser call: the sampler turns the logits into a clean
embedding estimate and moves the latent to the next noise level. The call
after the last noise level reads the tokens out, and one more causal pass
over those tokens commits them to the KV cache.
"""

from __future__ import annotations

import copy
import math
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
from torch import nn

from vllm.config import VllmConfig
from vllm.config.compilation import CUDAGraphMode
from vllm.logger import init_logger
from vllm.model_executor.models.mistral import MistralDecoderLayer, MistralModel
from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    WeightsMapper,
    maybe_prefix,
)
from vllm.utils.torch_utils import async_tensor_h2d
from vllm.v1.worker.gpu.attn_utils import build_attn_metadata
from vllm.v1.worker.gpu.model_states.interface import ModelState
from vllm.v1.worker.gpu.sample.output import SamplerOutput
from vllm.v1.worker.gpu.sample.penalties import use_penalty

logger = init_logger(__name__)

GAMMA_EMBED_DIM = 64
SAMPLERS = ("fresh_x0", "ddpm", "ddim")
TIME_SCHEDULES = ("edm_sigma", "log_snr")

PHASE_PREFILL, PHASE_DENOISE, PHASE_COMMIT = 0, 1, 2

# Bounds the transient `[rows, vocab]` fp32 logits and softmax of one chunk.
_MAX_LOGIT_ROWS = 1024

_UNUSED_WEIGHT_PREFIXES = ("ar_head.", "noise_schedule.")


def gamma_embed(gamma: torch.Tensor, dim: int = GAMMA_EMBED_DIM) -> torch.Tensor:
    """Sinusoidal features of the negative log-SNR `gamma`."""
    freqs = torch.exp(
        torch.linspace(-5.0, 5.0, dim // 2, dtype=torch.float32, device=gamma.device)
    )
    angles = freqs * gamma.float().unsqueeze(-1)
    return torch.cat([angles.sin(), angles.cos()], dim=-1)


def sigma_time_grid(
    steps: int,
    gamma_0: float,
    gamma_1: float,
    time_schedule: str = "edm_sigma",
    rho: float = 7.0,
) -> torch.Tensor:
    """`steps + 1` noise levels from `gamma_1` (noisy) down to `gamma_0`."""
    if time_schedule == "log_snr":
        return torch.linspace(gamma_1, gamma_0, steps + 1, dtype=torch.float64)
    if time_schedule != "edm_sigma":
        raise ValueError(f"time_schedule must be one of {TIME_SCHEDULES}")
    sigma_max = math.exp(gamma_1 / 2)
    sigma_min = math.exp(gamma_0 / 2)
    frac = torch.arange(steps + 1, dtype=torch.float64) / steps
    sigmas = (
        sigma_max ** (1 / rho)
        + frac * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))
    ) ** rho
    return 2 * torch.log(sigmas.clamp_min(1e-30))


@dataclass
class SigmaSchedule:
    """Per-step constants, each of shape `[steps + 1]` in fp64.

    Index `i` is the noise level of denoiser call `i`. The `next_*` entries
    are the level the latent moves to after that call; the last index is the
    readout call and never reads them.
    """

    steps: int
    gamma: torch.Tensor
    alpha: torch.Tensor
    sigma: torch.Tensor
    alpha_sq: torch.Tensor
    in_scale: torch.Tensor
    prior_scale: torch.Tensor
    next_alpha: torch.Tensor
    next_sigma: torch.Tensor
    next_alpha_sq: torch.Tensor
    ddpm_c: torch.Tensor

    @classmethod
    def build(cls, grid: torch.Tensor, embed_dim: int) -> SigmaSchedule:
        grid = grid.double()
        alpha_sq = torch.sigmoid(-grid)
        sigma_sq = torch.sigmoid(grid)
        alpha = alpha_sq.sqrt()
        sigma = sigma_sq.sqrt()
        nxt = torch.cat([grid[1:], grid[-1:]])
        next_alpha_sq = torch.sigmoid(-nxt)
        return cls(
            steps=grid.numel() - 1,
            gamma=grid,
            alpha=alpha,
            sigma=sigma,
            alpha_sq=alpha_sq,
            in_scale=(alpha_sq / embed_dim + sigma_sq).rsqrt(),
            prior_scale=alpha / sigma_sq,
            next_alpha=next_alpha_sq.sqrt(),
            next_sigma=torch.sigmoid(nxt).sqrt(),
            next_alpha_sq=next_alpha_sq,
            ddpm_c=-torch.expm1(nxt - grid),
        )

    def to(self, device: torch.device) -> SigmaSchedule:
        fields = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in self.__dict__.items()
        }
        return SigmaSchedule(**fields)


def sigma_update(
    z: torch.Tensor,
    x0: torch.Tensor,
    step: torch.Tensor,
    schedule: SigmaSchedule,
    sampler: str,
    score_temp: float,
    noise: torch.Tensor | None,
) -> torch.Tensor:
    """Move latents `z` (`[rows, embed_dim]`, fp64) to the next noise level.

    Args:
        z: Current latents.
        x0: The denoiser's clean-embedding estimate for the same rows.
        step: Denoiser-call index of each row, `[rows]`.
        schedule: Noise-level constants.
        sampler: One of `SAMPLERS`.
        score_temp: Divides the noise estimate; 1 leaves it unchanged.
        noise: Standard normal noise like `z`; unused by `ddim`.

    Returns:
        The latents at the next noise level.

    """
    x0 = x0.double()
    alpha_t = schedule.alpha[step].unsqueeze(-1)
    sigma_t = schedule.sigma[step].unsqueeze(-1)
    alpha_s = schedule.next_alpha[step].unsqueeze(-1)
    sigma_s = schedule.next_sigma[step].unsqueeze(-1)
    epsilon = (z - alpha_t * x0) / sigma_t.clamp_min(1e-8)
    if score_temp != 1.0:
        epsilon = epsilon / score_temp
        x0 = (z - sigma_t * epsilon) / alpha_t.clamp_min(1e-8)
    if sampler == "ddim":
        return alpha_s * x0 + sigma_s * epsilon
    assert noise is not None
    if sampler == "fresh_x0":
        return alpha_s * x0 + sigma_s * noise
    if sampler == "ddpm":
        c = schedule.ddpm_c[step].unsqueeze(-1)
        alpha_sq_t = schedule.alpha_sq[step].unsqueeze(-1)
        alpha_sq_s = schedule.next_alpha_sq[step].unsqueeze(-1)
        return (
            (1 - c) * (alpha_sq_s / alpha_sq_t).sqrt() * z
            + c * alpha_s * x0
            + (c * (1 - alpha_sq_s)).clamp_min(0).sqrt() * noise
        )
    raise ValueError(f"sampler must be one of {SAMPLERS}")


@dataclass
class SigmaGenerationSettings:
    steps: int
    sampler: str
    score_temp: float
    time_schedule: str
    time_rho: float
    last_step_sampling: str

    @classmethod
    def resolve(
        cls,
        hf_config: Any,
        gen_config: dict[str, Any],
        max_denoising_steps: int | None,
    ) -> SigmaGenerationSettings:
        """Few-step distilled checkpoints run at their native step count and
        sampler; other checkpoints follow the reference `generate` defaults."""
        native = getattr(hf_config, "native_family", None) == "phased_dmd"
        if native:
            steps = int(hf_config.native_nfe)
            if max_denoising_steps not in (None, steps):
                raise ValueError(
                    f"This checkpoint is distilled for {steps} denoising steps; "
                    f"max_denoising_steps={max_denoising_steps} is not supported."
                )
        else:
            steps = int(max_denoising_steps or gen_config.get("steps") or 32)
        sampler = gen_config.get("sampler") or ("fresh_x0" if native else "ddpm")
        if sampler not in SAMPLERS:
            raise ValueError(f"sampler must be one of {SAMPLERS}, got {sampler!r}")
        if sampler == "fresh_x0" and not native:
            raise ValueError("The fresh_x0 sampler needs a few-step checkpoint.")
        guidance = float(gen_config.get("guidance_scale", 1.0))
        if guidance != 1.0:
            raise ValueError(
                "Classifier-free guidance is not supported for Nemotron-Labs-Sigma "
                f"in vLLM (guidance_scale={guidance}); it needs a second, "
                "context-free denoiser pass per step."
            )
        time_schedule = gen_config.get("time_schedule", "edm_sigma")
        if time_schedule not in TIME_SCHEDULES:
            raise ValueError(f"time_schedule must be one of {TIME_SCHEDULES}")
        last = gen_config.get("last_step_sampling", "argmax")
        if last not in ("argmax", "softmax"):
            raise ValueError("last_step_sampling must be 'argmax' or 'softmax'")
        return cls(
            steps=steps,
            sampler=sampler,
            score_temp=float(gen_config.get("score_temp", 1.0 if native else 0.75)),
            time_schedule=time_schedule,
            time_rho=float(gen_config.get("time_rho", 7.0)),
            last_step_sampling=last,
        )


class _SigmaDecoderLayer(MistralDecoderLayer):
    def __init__(self, vllm_config: VllmConfig, prefix: str = "") -> None:
        hf_config = vllm_config.model_config.hf_config
        config = copy.copy(hf_config)
        config.rope_parameters = hf_config.backbone_rope_parameters()
        config.llama_4_scaling = hf_config.backbone_llama_4_scaling()
        super().__init__(vllm_config=vllm_config, prefix=prefix, config=config)


class SigmaLatentSpace(nn.Module):
    """The embedding geometry and the maps between latents and the backbone.

    The reference code runs this math in fp32/fp64 whatever the backbone
    dtype, so these small tensors stay fp32.
    """

    def __init__(self, config: Any):
        super().__init__()
        self.embed_dim = config.embed_dim
        self.disable_output_prior = config.replaid_disable_output_prior
        self.disable_time_conditioning = config.replaid_disable_time_conditioning

        def fp32(*shape: int) -> nn.Parameter:
            return nn.Parameter(
                torch.zeros(*shape, dtype=torch.float32), requires_grad=False
            )

        hidden = config.hidden_size
        self.replaid_embedding_matrix = nn.ParameterDict(
            {"matrix": fp32(config.vocab_size, config.embed_dim)}
        )
        self.replaid_input_linear = nn.ParameterDict(
            {"weight": fp32(hidden, config.embed_dim)}
        )
        self.replaid_selfcond_linear = (
            nn.ParameterDict({"weight": fp32(hidden, config.embed_dim)})
            if config.replaid_selfcond
            else None
        )
        self.replaid_gamma_linear = nn.ParameterDict(
            {"weight": fp32(hidden, GAMMA_EMBED_DIM)}
        )
        if config.schedule_parameterization == "projected_bernstein":
            bounds = {"endpoint_values": fp32(2)}
        else:
            bounds = {"gamma_0": fp32(), "gamma_1": fp32()}
        self.gamma_bounds = nn.ParameterDict(bounds)

        self._embedding: torch.Tensor | None = None
        self.schedule: SigmaSchedule | None = None
        self._time_embeds: torch.Tensor | None = None

    def reset_cache(self) -> None:
        self._embedding = self.schedule = self._time_embeds = None

    def gamma_endpoints(self) -> tuple[float, float]:
        bounds = self.gamma_bounds
        if "endpoint_values" in bounds:
            values = bounds["endpoint_values"].double()
            return float(values[0]), float(values[1])
        return float(bounds["gamma_0"].double()), float(bounds["gamma_1"].double())

    def embedding_matrix(self) -> torch.Tensor:
        """Row-normalized `[vocab, embed_dim]` embedding geometry."""
        if self._embedding is None:
            matrix = self.replaid_embedding_matrix["matrix"].float()
            self._embedding = matrix / (matrix.norm(dim=-1, keepdim=True) + 1e-8)
        return self._embedding

    def set_schedule(self, settings: SigmaGenerationSettings) -> SigmaSchedule:
        gamma_0, gamma_1 = self.gamma_endpoints()
        grid = sigma_time_grid(
            settings.steps, gamma_0, gamma_1, settings.time_schedule, settings.time_rho
        )
        weight = self.replaid_gamma_linear["weight"]
        self.schedule = SigmaSchedule.build(grid, self.embed_dim).to(weight.device)
        if self.disable_time_conditioning:
            self._time_embeds = None
        else:
            self._time_embeds = gamma_embed(self.schedule.gamma) @ weight.t()
        return self.schedule

    def noisy_inputs_embeds(
        self, z: torch.Tensor, x_selfcond: torch.Tensor, step: torch.Tensor
    ) -> torch.Tensor:
        """Backbone inputs for latents `z` (`[rows, embed_dim]`) at denoiser
        call `step` (`[rows]`). A zero `x_selfcond` adds nothing."""
        assert self.schedule is not None
        x_in = (z * self.schedule.in_scale[step].unsqueeze(-1)).float()
        h = x_in @ self.replaid_input_linear["weight"].t()
        if self.replaid_selfcond_linear is not None:
            scaled = x_selfcond * math.sqrt(self.embed_dim)
            h = h + scaled @ self.replaid_selfcond_linear["weight"].t()
        if self._time_embeds is not None:
            h = h + self._time_embeds[step]
        return h

    def add_output_prior(
        self, logits: torch.Tensor, z: torch.Tensor, step: torch.Tensor
    ) -> torch.Tensor:
        """Add the analytic prior of latents `z` to the head's fp32 logits."""
        assert self.schedule is not None
        if self.disable_output_prior:
            return logits
        scaled = (self.schedule.prior_scale[step].unsqueeze(-1) * z).float()
        return logits.addmm_(scaled, self.embedding_matrix().t())


class NemotronLabsSigmaForBlockDiffusion(nn.Module):
    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_prefix={
            "encoder.": "model.",
            "replaid_": "latent.replaid_",
            "gamma_bounds.": "latent.gamma_bounds.",
        }
    )

    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
    }

    @staticmethod
    def get_model_state_cls():
        return NemotronLabsSigmaModelState

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        self.config = config
        self.model = MistralModel(
            vllm_config=vllm_config,
            prefix=maybe_prefix(prefix, "model"),
            layer_type=_SigmaDecoderLayer,
        )
        self.diffusion_head = nn.Linear(
            config.hidden_size, config.vocab_size, bias=False
        )
        self.latent = SigmaLatentSpace(config)

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: Any | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        if intermediate_tensors is not None:
            inputs_embeds = None
        return self.model(input_ids, positions, intermediate_tensors, inputs_embeds)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.diffusion_head(hidden_states).float()

    def denoise_logits(
        self, hidden_states: torch.Tensor, z: torch.Tensor, step: torch.Tensor
    ) -> torch.Tensor:
        return self.latent.add_output_prior(self.compute_logits(hidden_states), z, step)

    def is_unused_checkpoint_weight(self, name: str) -> bool:
        return name.startswith(_UNUSED_WEIGHT_PREFIXES)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(
            self, ignore_unexpected_prefixes=list(_UNUSED_WEIGHT_PREFIXES)
        )
        loaded = loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)
        self.latent.reset_cache()
        return loaded


class NemotronLabsSigmaRequestStates:
    """Per-request block state, indexed by request slot.

    The schedule is fixed, so phase and step live on the host: no step ever
    reads device state back to decide what to run.
    """

    def __init__(
        self,
        max_num_reqs: int,
        canvas_length: int,
        embed_dim: int,
        device: torch.device,
        seed: int | None = None,
    ):
        self.max_num_reqs = max_num_reqs
        self.canvas_length = canvas_length
        self.embed_dim = embed_dim
        self.device = device

        self.z = torch.zeros(
            max_num_reqs, canvas_length, embed_dim, dtype=torch.float64, device=device
        )
        self.x_selfcond = torch.zeros(
            max_num_reqs, canvas_length, embed_dim, dtype=torch.float32, device=device
        )
        self.canvas = torch.zeros(
            max_num_reqs, canvas_length, dtype=torch.int64, device=device
        )
        self.is_causal = torch.ones(max_num_reqs, dtype=torch.bool, device=device)
        self.phase_np = np.full(max_num_reqs, PHASE_PREFILL, dtype=np.int8)
        self.step_np = np.zeros(max_num_reqs, dtype=np.int32)

        self._generator = torch.Generator(device=device)
        if seed is not None:
            self._generator.manual_seed(seed)
        else:
            self._generator.seed()

    def randn(self, *shape: int) -> torch.Tensor:
        return torch.randn(
            *shape, dtype=torch.float64, device=self.device, generator=self._generator
        )

    def _slots(self, slots_np: np.ndarray) -> torch.Tensor:
        return async_tensor_h2d(slots_np.astype(np.int64), device=self.device)

    def add_request(self, slot_idx: int) -> None:
        self.phase_np[slot_idx] = PHASE_PREFILL
        self.step_np[slot_idx] = 0
        self.is_causal[slot_idx].fill_(True)

    def remove_request(self, slot_idx: int) -> None:
        self.add_request(slot_idx)

    def start_block(self, slots_np: np.ndarray) -> None:
        """Give `slots_np` a fresh block of noise at the first noise level."""
        if len(slots_np) == 0:
            return
        slots = self._slots(slots_np)
        self.z[slots] = self.randn(len(slots_np), self.canvas_length, self.embed_dim)
        self.x_selfcond.index_fill_(0, slots, 0)
        self.canvas.index_fill_(0, slots, 0)
        self.is_causal.index_fill_(0, slots, False)
        self.phase_np[slots_np] = PHASE_DENOISE
        self.step_np[slots_np] = 0

    def start_commit(self, slots_np: np.ndarray) -> None:
        if len(slots_np) == 0:
            return
        self.is_causal.index_fill_(0, self._slots(slots_np), True)
        self.phase_np[slots_np] = PHASE_COMMIT


def _canvas_rows(
    starts_np: np.ndarray, lens_np: np.ndarray, slots_np: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Flat rows of several requests' canvases.

    Returns each row's index (`starts_np[i] + offset`), request slot and
    canvas offset, for request `i` spanning `lens_np[i]` rows.
    """
    total = int(lens_np.sum())
    if total == 0:
        empty = np.zeros(0, dtype=np.int64)
        return empty, empty, empty
    req = np.repeat(np.arange(len(lens_np)), lens_np)
    offsets = np.arange(total) - np.repeat(np.cumsum(lens_np) - lens_np, lens_np)
    rows = starts_np[req] + offsets
    return rows.astype(np.int64), slots_np[req].astype(np.int64), offsets


def write_noisy_inputs(
    inputs_embeds: torch.Tensor,
    input_batch: Any,
    states: NemotronLabsSigmaRequestStates,
    latent: SigmaLatentSpace,
) -> None:
    """Overwrite the token embeddings of every denoising canvas in the batch
    with the embeddings of its latent. Commit passes keep their tokens."""
    num_reqs = input_batch.num_reqs
    slots_np = input_batch.idx_mapping_np[:num_reqs]
    num_logits_np = np.diff(input_batch.cu_num_logits_np[: num_reqs + 1])
    denoise_np = np.where(
        (num_logits_np > 0) & (states.phase_np[slots_np] == PHASE_DENOISE)
    )[0]
    starts_np = input_batch.query_start_loc_np[: num_reqs + 1]
    rows_np, row_slots_np, row_offsets_np = _canvas_rows(
        starts_np[denoise_np], np.diff(starts_np)[denoise_np], slots_np[denoise_np]
    )
    if len(rows_np) == 0:
        return
    device = inputs_embeds.device
    rows = async_tensor_h2d(rows_np, device=device)
    row_slots = async_tensor_h2d(row_slots_np, device=device)
    row_offsets = async_tensor_h2d(row_offsets_np, device=device)
    steps = async_tensor_h2d(
        states.step_np[row_slots_np].astype(np.int64), device=device
    )
    inputs_embeds[rows] = latent.noisy_inputs_embeds(
        states.z[row_slots, row_offsets],
        states.x_selfcond[row_slots, row_offsets],
        steps,
    ).to(inputs_embeds.dtype)


class NemotronLabsSigmaModelState(ModelState):
    num_new_sampled_tokens_per_step: int = 0

    def __init__(
        self,
        vllm_config: VllmConfig,
        model: nn.Module,
        encoder_cache: Any,
        device: torch.device,
    ) -> None:
        super().__init__(vllm_config, model, encoder_cache, device)
        hf_config = self.model_config.hf_config
        diffusion_config = vllm_config.diffusion_config
        self.canvas_length = (
            diffusion_config.canvas_length
            if diffusion_config is not None
            else hf_config.block_size
        )
        self.settings = SigmaGenerationSettings.resolve(
            hf_config,
            {
                **self.model_config.try_get_generation_config(),
                **self.model_config.override_generation_config,
            },
            diffusion_config.max_denoising_steps if diffusion_config else None,
        )
        self.diffusion_states = NemotronLabsSigmaRequestStates(
            max_num_reqs=self.max_num_reqs,
            canvas_length=self.canvas_length,
            embed_dim=hf_config.embed_dim,
            device=device,
            seed=self.model_config.seed,
        )
        self._req_id_to_index: dict[str, int] = {}

        # Persistent buffers so a FULL CUDA graph replays against the
        # addresses it captured; `_causal_buf` must be int32 for FlashAttention.
        self._causal_buf = torch.zeros(
            self.max_num_reqs, dtype=torch.int32, device=device
        )
        self._inputs_embeds_buf = torch.zeros(
            self.max_num_tokens,
            hf_config.hidden_size,
            dtype=self.model_config.dtype,
            device=device,
        )

    def get_supported_generation_tasks(self):
        return ("generate",)

    def compute_sample_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return hidden_states

    def dummy_sampler_run(self, sampler: Any, hidden_states: torch.Tensor) -> bool:
        sampler.profile_run(hidden_states)
        return True

    def custom_sampler(self, sampler: Any) -> tuple[Any, Any] | None:
        latent = self.model.latent
        return NemotronLabsSigmaSampler(
            sampler=sampler,
            states=self.diffusion_states,
            schedule=latent.set_schedule(self.settings),
            settings=self.settings,
            denoise_logits=self.model.denoise_logits,
            embedding_matrix=latent.embedding_matrix(),
        ), None

    def add_request(self, req_index: int, new_req_data: Any) -> None:
        self._req_id_to_index[new_req_data.req_id] = req_index
        self.diffusion_states.add_request(req_index)

    def remove_request(self, req_id: str) -> None:
        idx = self._req_id_to_index.pop(req_id, None)
        if idx is not None:
            self.diffusion_states.remove_request(idx)

    def prepare_inputs_embeds(
        self, scheduled_encoder_inputs, input_batch, req_states
    ) -> torch.Tensor | None:
        return None

    def prepare_inputs(self, input_batch, req_states) -> dict[str, Any]:
        num_tokens = input_batch.num_tokens
        inputs_embeds = self._inputs_embeds_buf[: input_batch.num_tokens_after_padding]
        inputs_embeds[:num_tokens].copy_(
            self.model.embed_input_ids(input_batch.input_ids[:num_tokens])
        )

        if input_batch.num_draft_tokens > 0 and self._req_id_to_index:
            write_noisy_inputs(
                inputs_embeds, input_batch, self.diffusion_states, self.model.latent
            )

        return {"inputs_embeds": inputs_embeds}

    def prepare_dummy_inputs(self, num_reqs: int, num_tokens: int) -> dict[str, Any]:
        return {"inputs_embeds": self._inputs_embeds_buf[:num_tokens]}

    def prepare_attn(
        self,
        input_batch,
        cudagraph_mode,
        block_tables,
        slot_mappings,
        attn_groups,
        kv_cache_config,
        for_capture=False,
        ubatch_idx: int = 0,
    ) -> dict[str, Any]:
        assert ubatch_idx == 0, "DBO is not supported"
        if cudagraph_mode == CUDAGraphMode.FULL:
            num_reqs = input_batch.num_reqs_after_padding
            num_tokens = input_batch.num_tokens_after_padding
        else:
            num_reqs = input_batch.num_reqs
            num_tokens = input_batch.num_tokens

        actual_num_reqs = input_batch.num_reqs
        slots = input_batch.idx_mapping[:actual_num_reqs]
        self._causal_buf[:actual_num_reqs].copy_(self.diffusion_states.is_causal[slots])
        if actual_num_reqs < num_reqs:
            self._causal_buf[actual_num_reqs:num_reqs] = 0

        return build_attn_metadata(
            attn_groups=attn_groups,
            num_reqs=num_reqs,
            num_tokens=num_tokens,
            query_start_loc_gpu=input_batch.query_start_loc,
            query_start_loc_cpu=torch.from_numpy(input_batch.query_start_loc_np),
            max_query_len=input_batch.num_scheduled_tokens.max().item(),
            seq_lens=input_batch.seq_lens,
            max_seq_len=self.max_model_len,
            block_tables=block_tables,
            slot_mappings=slot_mappings,
            kv_cache_config=kv_cache_config,
            causal=self._causal_buf[:num_reqs],
        )


_NO_PENALTIES_STATE = SimpleNamespace(output_bin_counts=None)


class NemotronLabsSigmaSampler:
    """Advances every scheduled block by one denoiser call.

    A request cycles through `steps` latent updates, one readout call and one
    causal commit pass, which emits the block's tokens.
    """

    def __init__(
        self,
        sampler: Any,
        states: NemotronLabsSigmaRequestStates,
        schedule: SigmaSchedule,
        settings: SigmaGenerationSettings,
        denoise_logits: Callable[
            [torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor
        ],
        embedding_matrix: torch.Tensor,
    ):
        self.sampling_states = sampler.sampling_states
        self.logprob_token_ids_state = sampler.logprob_token_ids_state
        self.req_states = sampler.req_states
        self.states = states
        self.schedule = schedule
        self.settings = settings
        self.denoise_logits = denoise_logits
        self.embedding_matrix = embedding_matrix
        self.canvas_length = states.canvas_length

        device = states.device
        self._sampled = torch.zeros(
            states.max_num_reqs, self.canvas_length, dtype=torch.int32, device=device
        )
        self._num_sampled = torch.zeros(
            states.max_num_reqs, dtype=torch.int32, device=device
        )

    def add_request(self, req_idx: int, sampling_params: Any) -> None:
        if use_penalty(sampling_params):
            logger.warning_once(
                "Nemotron-Labs-Sigma does not support repetition/frequency/"
                "presence penalties; ignoring them for this request."
            )
        self.sampling_states.add_request(req_idx, sampling_params)
        self.logprob_token_ids_state.add_request(req_idx, sampling_params)

    def apply_staged_writes(self) -> None:
        self.sampling_states.apply_staged_writes()
        self.logprob_token_ids_state.apply_staged_writes()

    @property
    def penalties_state(self):
        return _NO_PENALTIES_STATE

    @torch.inference_mode()
    def profile_run(self, hidden_states: torch.Tensor) -> None:
        CL = self.canvas_length
        device = hidden_states.device
        bounds = np.array([0, CL], dtype=np.int32)
        dummy_batch = SimpleNamespace(
            num_reqs=1,
            num_draft_tokens=CL,
            idx_mapping_np=np.zeros(1, dtype=np.int32),
            idx_mapping=torch.zeros(1, dtype=torch.int64, device=device),
            cu_num_logits_np=bounds,
            query_start_loc_np=bounds,
            query_start_loc=torch.tensor(bounds, device=device),
        )
        self.states.start_block(np.zeros(1, dtype=np.int64))
        self(hidden_states.new_zeros((CL, hidden_states.shape[-1])), dummy_batch)
        self.states.add_request(0)

    def _finish_prefills(
        self, input_batch: Any, prefill_indices_np: np.ndarray
    ) -> None:
        """Start the first block of every request whose prompt completes this
        step. A prompt spanning several chunks stays causal until its last."""
        done_np = (
            input_batch.num_computed_prefill_tokens_np[prefill_indices_np]
            + input_batch.num_scheduled_tokens[prefill_indices_np]
            >= input_batch.prefill_len_np[prefill_indices_np]
        )
        self.states.start_block(input_batch.idx_mapping_np[prefill_indices_np[done_np]])

    def _output(
        self, input_batch: Any, num_logits_np: np.ndarray, width: int
    ) -> SamplerOutput:
        num_reqs = input_batch.num_reqs
        num_sampled = self._num_sampled[:num_reqs]
        num_logits = async_tensor_h2d(
            num_logits_np.astype(np.int32), device=num_sampled.device
        )
        starts = input_batch.query_start_loc[: num_reqs + 1]
        # A denoiser call commits nothing, so its whole canvas is rejected.
        num_rejected = torch.where(
            (num_logits > 0) & (num_sampled == 0),
            (starts[1:] - starts[:-1]).to(num_sampled.dtype),
            num_logits - num_sampled,
        )
        return SamplerOutput(
            sampled_token_ids=self._sampled[:num_reqs, :width],
            logprobs_tensors=None,
            num_nans=None,
            num_sampled=num_sampled,
            num_rejected=num_rejected,
        )

    def _readout(self, logits: torch.Tensor) -> torch.Tensor:
        if self.settings.last_step_sampling == "softmax":
            probs = torch.softmax(logits, dim=-1)
            return torch.multinomial(
                probs, 1, generator=self.states._generator
            ).squeeze(-1)
        return logits.argmax(dim=-1)

    def _denoise(
        self,
        hidden_states: torch.Tensor,
        rows_np: np.ndarray,
        row_slots_np: np.ndarray,
        row_offsets_np: np.ndarray,
    ) -> None:
        """Run one denoiser call's sampling for the given canvas rows: update
        the latent, or read the tokens out on a request's last call."""
        states = self.states
        device = hidden_states.device
        steps_np = states.step_np[row_slots_np].astype(np.int64)
        is_readout_np = steps_np == self.schedule.steps
        for start in range(0, len(rows_np), _MAX_LOGIT_ROWS):
            chunk = slice(start, start + _MAX_LOGIT_ROWS)
            rows = async_tensor_h2d(rows_np[chunk], device=device)
            slots = async_tensor_h2d(row_slots_np[chunk], device=device)
            offsets = async_tensor_h2d(row_offsets_np[chunk], device=device)
            steps = async_tensor_h2d(steps_np[chunk], device=device)
            z = states.z[slots, offsets]
            logits = self.denoise_logits(hidden_states[rows], z, steps)

            readout_np = np.flatnonzero(is_readout_np[chunk])
            if len(readout_np) == len(rows):
                states.canvas[slots, offsets] = self._readout(logits)
                continue
            if len(readout_np) > 0:
                sel = async_tensor_h2d(readout_np, device=device)
                states.canvas[slots[sel], offsets[sel]] = self._readout(logits[sel])

            cond_mean = torch.softmax(logits, dim=-1) @ self.embedding_matrix
            noise = (
                None
                if self.settings.sampler == "ddim"
                else states.randn(len(rows), states.embed_dim)
            )
            # Readout rows also get a latent here, from the schedule's
            # repeated last level. Nothing reads it: their block ends.
            states.z[slots, offsets] = sigma_update(
                z,
                cond_mean,
                steps,
                self.schedule,
                self.settings.sampler,
                self.settings.score_temp,
                noise,
            )
            states.x_selfcond[slots, offsets] = cond_mean

    def __call__(
        self,
        hidden_states: torch.Tensor,
        input_batch: Any,
        draft_logits: torch.Tensor | None = None,
    ) -> SamplerOutput:
        states = self.states
        num_reqs = input_batch.num_reqs
        device = hidden_states.device
        slots_np = input_batch.idx_mapping_np[:num_reqs]
        self._num_sampled[:num_reqs].zero_()

        if input_batch.num_draft_tokens == 0:
            self._finish_prefills(input_batch, np.arange(num_reqs))
            self._sampled[:num_reqs, :1].zero_()
            return self._output(input_batch, np.zeros(num_reqs, dtype=np.int32), 1)

        num_logits_np = np.diff(input_batch.cu_num_logits_np[: num_reqs + 1])
        decode_np = np.where(num_logits_np > 0)[0]
        prefill_np = np.where(num_logits_np == 0)[0]
        if len(prefill_np) > 0:
            self._finish_prefills(input_batch, prefill_np)
        self._sampled[:num_reqs].zero_()

        lens_np = num_logits_np[decode_np]
        row_starts_np = np.cumsum(lens_np) - lens_np
        decode_slots_np = slots_np[decode_np]
        phases_np = states.phase_np[decode_slots_np]

        # Commit passes ran causally over the block's tokens: emit them and
        # start the next block.
        commit_np = np.flatnonzero(phases_np == PHASE_COMMIT)
        if len(commit_np) > 0:
            commit_slots_np = decode_slots_np[commit_np]
            commit_idx = async_tensor_h2d(
                decode_np[commit_np].astype(np.int64), device=device
            )
            commit_slots = async_tensor_h2d(
                commit_slots_np.astype(np.int64), device=device
            )
            self._sampled[commit_idx] = states.canvas[commit_slots].to(
                self._sampled.dtype
            )
            self._num_sampled[commit_idx] = async_tensor_h2d(
                lens_np[commit_np].astype(np.int32), device=device
            )

        denoise_np = np.flatnonzero(phases_np == PHASE_DENOISE)
        if len(denoise_np) > 0:
            denoise_slots_np = decode_slots_np[denoise_np]
            self._denoise(
                hidden_states,
                *_canvas_rows(
                    row_starts_np[denoise_np], lens_np[denoise_np], denoise_slots_np
                ),
            )
            was_readout_np = states.step_np[denoise_slots_np] == self.schedule.steps
            states.step_np[denoise_slots_np[~was_readout_np]] += 1
            states.start_commit(denoise_slots_np[was_readout_np])

        if len(commit_np) > 0:
            states.start_block(commit_slots_np)

        all_slots = input_batch.idx_mapping[:num_reqs]
        self.req_states.draft_tokens[all_slots, : self.canvas_length] = states.canvas[
            all_slots
        ]
        return self._output(input_batch, num_logits_np, self.canvas_length)
