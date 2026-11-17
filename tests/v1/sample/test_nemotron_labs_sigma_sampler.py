# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Nemotron-Labs-Sigma's engine-step sampler must generate the same tokens as
the checkpoint's own blockwise loop: `steps` latent updates and one readout
per block, each block conditioned on the tokens committed before it."""

import math
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vllm.exceptions import VLLMValidationError
from vllm.model_executor.models import nemotron_labs_sigma as sigma
from vllm.sampling_params import SamplingParams
from vllm.transformers_utils.configs import NemotronLabsSigmaConfig
from vllm.utils.diffusion import validate_continuous_diffusion_sampling_params

VOCAB, HIDDEN, EMBED, BLOCK = 50, 24, 8, 4
GAMMA_0, GAMMA_1 = -3.0, 6.0


def _latent(selfcond: bool = True) -> sigma.SigmaLatentSpace:
    torch.manual_seed(0)
    latent = sigma.SigmaLatentSpace(
        SimpleNamespace(
            vocab_size=VOCAB,
            hidden_size=HIDDEN,
            embed_dim=EMBED,
            replaid_selfcond=selfcond,
            replaid_disable_output_prior=False,
            replaid_disable_time_conditioning=False,
            schedule_parameterization="projected_bernstein",
        )
    )
    for param in latent.parameters():
        param.copy_(torch.randn_like(param))
    latent.gamma_bounds["endpoint_values"].copy_(torch.tensor([GAMMA_0, GAMMA_1]))
    # Keep the time signal, shared by every position, from saturating the toy.
    latent.replaid_gamma_linear["weight"].mul_(0.05)
    return latent


class _ToyDenoiser:
    """Stands in for the backbone and head: every canvas position sees the
    whole canvas (bidirectional) and the tokens committed so far (context)."""

    def __init__(self):
        g = torch.Generator().manual_seed(1)
        self.mix = torch.randn(HIDDEN, HIDDEN, generator=g) / math.sqrt(HIDDEN)
        self.token_embeds = torch.randn(VOCAB, HIDDEN, generator=g)
        self.head = torch.randn(VOCAB, HIDDEN, generator=g)

    def __call__(self, inputs_embeds: torch.Tensor, context: list[int]):
        ctx = self.token_embeds[context].mean(dim=0)
        shared = inputs_embeds.mean(dim=0) + ctx
        return torch.tanh(0.5 * inputs_embeds @ self.mix + 0.1 * shared)

    def logits(self, hidden: torch.Tensor) -> torch.Tensor:
        return hidden @ self.head.t()


def _reference_block(
    latent, toy, z, context, steps, sampler, score_temp, time_schedule, noise=None
):
    """One block of the checkpoint's `generate`, written out step by step."""
    embedding = latent.replaid_embedding_matrix["matrix"]
    embedding = embedding / (embedding.norm(dim=-1, keepdim=True) + 1e-8)
    if time_schedule == "log_snr":
        grid = [GAMMA_1 + i / steps * (GAMMA_0 - GAMMA_1) for i in range(steps + 1)]
    else:
        hi, lo = math.exp(GAMMA_1 / 2) ** (1 / 7), math.exp(GAMMA_0 / 2) ** (1 / 7)
        grid = [
            2 * math.log((hi + i / steps * (lo - hi)) ** 7) for i in range(steps + 1)
        ]
    freqs = torch.exp(torch.linspace(-5.0, 5.0, 32))

    x_selfcond = None
    for i, gamma in enumerate(grid):
        alpha_sq, sigma_sq = 1 / (1 + math.exp(gamma)), 1 / (1 + math.exp(-gamma))
        alpha = math.sqrt(alpha_sq)
        x_in = (z / math.sqrt(alpha_sq / EMBED + sigma_sq)).float()
        h = x_in @ latent.replaid_input_linear["weight"].t()
        if x_selfcond is not None:
            h = (
                h
                + (x_selfcond * math.sqrt(EMBED))
                @ latent.replaid_selfcond_linear["weight"].t()
            )
        angles = freqs * gamma
        features = torch.cat([angles.sin(), angles.cos()])
        h = h + features @ latent.replaid_gamma_linear["weight"].t()

        logits = toy.logits(toy(h, context)).float()
        logits = logits + ((alpha / sigma_sq) * z).float() @ embedding.t()
        if i == steps:
            return logits.argmax(dim=-1).tolist()
        if i == 0 and noise is not None:
            return _reference_update(
                latent, z, logits, gamma, grid[1], sampler, score_temp, noise
            )

        cond_mean = torch.softmax(logits, dim=-1) @ embedding
        if latent.replaid_selfcond_linear is not None:
            x_selfcond = cond_mean
        z = _reference_update(
            latent, z, logits, gamma, grid[i + 1], sampler, score_temp, None
        )


def _reference_update(latent, z, logits, gamma, gamma_s, sampler, score_temp, noise):
    embedding = latent.embedding_matrix()
    alpha_sq, alpha_sq_s = 1 / (1 + math.exp(gamma)), 1 / (1 + math.exp(gamma_s))
    alpha, sigma_t = math.sqrt(alpha_sq), math.sqrt(1 - alpha_sq)
    alpha_s, sigma_s = math.sqrt(alpha_sq_s), math.sqrt(1 - alpha_sq_s)
    x0 = (torch.softmax(logits, dim=-1) @ embedding).double()
    epsilon = (z - alpha * x0) / sigma_t
    if score_temp != 1.0:
        epsilon = epsilon / score_temp
        x0 = (z - sigma_t * epsilon) / alpha
    if sampler == "ddim":
        return alpha_s * x0 + sigma_s * epsilon
    if sampler == "fresh_x0":
        return alpha_s * x0 + sigma_s * noise
    c = -math.expm1(gamma_s - gamma)
    return (
        (1 - c) * math.sqrt(alpha_sq_s / alpha_sq) * z
        + c * alpha_s * x0
        + math.sqrt(max(c * (1 - alpha_sq_s), 0)) * noise
    )


class _Engine:
    """The part of the runner the sampler talks to, for a few requests that
    each occupy one slot."""

    def __init__(self, latent, toy, settings, prompts, num_slots):
        self.latent, self.toy, self.prompts = latent, toy, prompts
        self.states = sigma.NemotronLabsSigmaRequestStates(
            num_slots, BLOCK, EMBED, torch.device("cpu"), seed=0
        )
        self.block_noise: dict[int, list[torch.Tensor]] = {s: [] for s in prompts}
        self.req_states = SimpleNamespace(
            draft_tokens=torch.zeros(num_slots, BLOCK, dtype=torch.int64)
        )
        base = SimpleNamespace(
            sampling_states=None,
            logprob_token_ids_state=None,
            req_states=self.req_states,
        )
        self.sampler = sigma.NemotronLabsSigmaSampler(
            sampler=base,
            states=self.states,
            schedule=latent.set_schedule(settings),
            settings=settings,
            denoise_logits=lambda h, z, step: latent.add_output_prior(
                toy.logits(h).float(), z, step
            ),
            embedding_matrix=latent.embedding_matrix(),
        )
        self.outputs: dict[int, list[int]] = {s: [] for s in prompts}
        self.prefilled: set[int] = set()
        self.commit_inputs: dict[int, list[list[int]]] = {s: [] for s in prompts}
        for slot in prompts:
            self.states.add_request(slot)

    def step(self, slots: list[int]) -> None:
        is_prefill = [s not in self.prefilled for s in slots]
        lens = [len(self.prompts[s]) if p else BLOCK for s, p in zip(slots, is_prefill)]
        starts = np.concatenate([[0], np.cumsum(lens)]).astype(np.int32)
        num_logits = np.array([0 if p else BLOCK for p in is_prefill])
        prompt_lens = np.array([len(self.prompts[s]) for s in slots])
        batch = SimpleNamespace(
            num_reqs=len(slots),
            num_draft_tokens=0 if all(is_prefill) else BLOCK,
            idx_mapping_np=np.array(slots, dtype=np.int64),
            idx_mapping=torch.tensor(slots, dtype=torch.int64),
            cu_num_logits_np=np.concatenate([[0], np.cumsum(num_logits)]),
            query_start_loc_np=starts,
            query_start_loc=torch.from_numpy(starts),
            num_computed_prefill_tokens_np=np.where(is_prefill, 0, prompt_lens),
            num_scheduled_tokens=np.array(lens),
            prefill_len_np=prompt_lens,
        )

        input_ids = [
            self.prompts[s] if p else self.req_states.draft_tokens[s].tolist()
            for s, p in zip(slots, is_prefill)
        ]
        inputs_embeds = self.toy.token_embeds[sum(input_ids, [])].clone()
        token_embeds = inputs_embeds.clone()
        sigma.write_noisy_inputs(inputs_embeds, batch, self.states, self.latent)

        hidden = []
        for i, (slot, prefill) in enumerate(zip(slots, is_prefill)):
            if prefill:
                continue
            rows = slice(starts[i], starts[i + 1])
            if self.states.phase_np[slot] == sigma.PHASE_COMMIT:
                # A commit pass must run over the block's own tokens.
                assert torch.equal(inputs_embeds[rows], token_embeds[rows])
                self.commit_inputs[slot].append(input_ids[i])
            context = self.prompts[slot] + self.outputs[slot]
            hidden.append(self.toy(inputs_embeds[rows], context))
        hidden_states = torch.cat(hidden) if hidden else torch.zeros(0, HIDDEN)

        out = self.sampler(hidden_states, batch)
        for i, slot in enumerate(slots):
            n = int(out.num_sampled[i])
            self.outputs[slot] += out.sampled_token_ids[i, :n].tolist()
            assert int(out.num_rejected[i]) == (0 if n or is_prefill[i] else BLOCK)
        self.prefilled.update(slots)
        for slot in slots:
            starting = self.states.phase_np[slot] == sigma.PHASE_DENOISE
            if starting and self.states.step_np[slot] == 0:
                self.block_noise[slot].append(self.states.z[slot].clone())


def _settings(steps, sampler, score_temp=1.0, time_schedule="edm_sigma"):
    return sigma.SigmaGenerationSettings(
        steps=steps,
        sampler=sampler,
        score_temp=score_temp,
        time_schedule=time_schedule,
        time_rho=7.0,
        last_step_sampling="argmax",
    )


@pytest.mark.parametrize(
    ("score_temp", "time_schedule", "selfcond"),
    [(1.0, "edm_sigma", True), (0.75, "log_snr", True), (0.75, "edm_sigma", False)],
)
def test_engine_steps_match_reference_generate(score_temp, time_schedule, selfcond):
    """Two requests admitted two steps apart, so every batch mixes prefill,
    denoise, readout and commit rows at different noise levels. DDIM draws no
    noise after the block's first latent, so the tokens are exact."""
    sampler, steps, num_blocks = "ddim", 3, 3
    latent, toy = _latent(selfcond), _ToyDenoiser()
    prompts = {0: [3, 7, 11], 2: [5, 9]}
    engine = _Engine(
        latent, toy, _settings(steps, sampler, score_temp, time_schedule), prompts, 4
    )

    # Prefill, then `steps` updates, one readout and one commit per block.
    steps_per_request = 1 + num_blocks * (steps + 2)
    for t in range(steps_per_request + 2):
        slots = [s for s, t0 in ((0, 0), (2, 2)) if 0 <= t - t0 < steps_per_request]
        engine.step(slots)

    for slot, prompt in prompts.items():
        expected: list[int] = []
        for block in range(num_blocks):
            expected += _reference_block(
                latent,
                toy,
                engine.block_noise[slot][block],
                prompt + expected,
                steps,
                sampler,
                score_temp,
                time_schedule,
            )
        assert engine.outputs[slot] == expected
        blocks = [expected[i : i + BLOCK] for i in range(0, len(expected), BLOCK)]
        assert engine.commit_inputs[slot] == blocks
        # Guards the toy denoiser against a degenerate, trivially equal output.
        assert len(set(expected)) > 2


@pytest.mark.parametrize("sampler", ["fresh_x0", "ddpm"])
@pytest.mark.parametrize("score_temp", [1.0, 0.75])
def test_stochastic_update_matches_reference(sampler, score_temp):
    latent, toy = _latent(), _ToyDenoiser()
    schedule = latent.set_schedule(_settings(3, sampler, score_temp))
    g = torch.Generator().manual_seed(2)
    z = torch.randn(BLOCK, EMBED, dtype=torch.float64, generator=g)
    noise = torch.randn(BLOCK, EMBED, dtype=torch.float64, generator=g)
    step = torch.zeros(BLOCK, dtype=torch.int64)

    hidden = toy(latent.noisy_inputs_embeds(z, torch.zeros(BLOCK, EMBED), step), [1])
    logits = latent.add_output_prior(toy.logits(hidden).float(), z, step)
    x0 = torch.softmax(logits, dim=-1) @ latent.embedding_matrix()
    actual = sigma.sigma_update(z, x0, step, schedule, sampler, score_temp, noise)

    expected = _reference_block(
        latent, toy, z, [1], 3, sampler, score_temp, "edm_sigma", noise
    )
    # The clean estimate is an fp32 softmax in both paths.
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


def test_time_grid_runs_from_noisy_to_clean_endpoint():
    for schedule in sigma.TIME_SCHEDULES:
        grid = sigma.sigma_time_grid(4, GAMMA_0, GAMMA_1, schedule)
        assert grid.dtype == torch.float64
        assert grid[0].item() == pytest.approx(GAMMA_1)
        assert grid[-1].item() == pytest.approx(GAMMA_0)
        assert (grid[1:] < grid[:-1]).all()


def test_distilled_checkpoint_runs_at_its_native_settings():
    """A few-step checkpoint only works at the step count it was distilled
    for, and guidance would need a pass the engine does not run."""
    config = NemotronLabsSigmaConfig(native_family="phased_dmd", native_nfe=4)
    settings = sigma.SigmaGenerationSettings.resolve(config, {}, None)
    assert (settings.steps, settings.sampler, settings.score_temp) == (
        4,
        "fresh_x0",
        1.0,
    )
    with pytest.raises(ValueError, match="distilled for 4"):
        sigma.SigmaGenerationSettings.resolve(config, {}, 8)
    with pytest.raises(ValueError, match="guidance"):
        sigma.SigmaGenerationSettings.resolve(config, {"guidance_scale": 2}, None)

    base = NemotronLabsSigmaConfig()
    settings = sigma.SigmaGenerationSettings.resolve(base, {}, None)
    assert (settings.steps, settings.sampler, settings.score_temp) == (32, "ddpm", 0.75)


def test_config_maps_checkpoint_rope_to_backbone_rope():
    """The checkpoint interpolates RoPE by its context ratio, not by the
    nominal YaRN factor, which only sets the attention scale."""
    config = NemotronLabsSigmaConfig()
    assert config.canvas_length == config.block_size == 32
    rope = config.backbone_rope_parameters()
    assert rope["factor"] == 4096 / 16384
    assert rope["attention_factor"] == 1.0
    assert config.backbone_llama_4_scaling() == {
        "original_max_position_embeddings": 16384,
        "beta": 0.1,
    }


@pytest.mark.parametrize(
    "params",
    [
        SamplingParams(logprobs=1),
        SamplingParams(extra_args={"diffusion_seed_canvas": [1, 2]}),
    ],
)
def test_requests_for_token_canvas_features_are_rejected(params):
    with pytest.raises(VLLMValidationError):
        validate_continuous_diffusion_sampling_params(params)
    validate_continuous_diffusion_sampling_params(SamplingParams(max_tokens=8))
