"""
Debug test script for self-distillation training.

This script runs a minimal training loop with detailed, color-coded output
and interactive pauses to verify the self-distillation pipeline is working correctly.

Key debug features:
1. Color-coded student input (yellow=prompt, green=generated tokens)
2. Color-coded proxy teacher input (same color scheme)
3. Per-token KL loss printing
4. Interactive pauses after each section
5. Gradient flow verification

Usage:
    conda activate tinker
    cd /home/devvrit03/tinker-cookbook-custom
    python -m tinker_cookbook.distillation.debug_self_distillation
"""

import asyncio
import logging
import os
import sys
from functools import partial
from typing import Sequence

import torch
import tinker
from termcolor import colored

from tinker_cookbook import renderers
from tinker_cookbook.distillation.self_distillation_datasets import (
    SelfDistillationEnv,
    DEFAULT_STUDENT_SUFFIX,
    DEFAULT_PROXY_TEACHER_TEMPLATE,
)
from tinker_cookbook.rl.problem_env import ProblemGroupBuilder
from tinker_cookbook.rl.types import EnvGroupBuilder, TrajectoryGroup
from tinker_cookbook.rl.train import do_group_rollout_and_filter_constant_reward
from tinker_cookbook.rl.data_processing import assemble_training_data, compute_advantages
from tinker_cookbook.tokenizer_utils import Tokenizer, get_tokenizer
from tinker_cookbook.utils.format_colorized import format_colorized

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# =============================================================================
# DEBUG CONFIGURATION
# =============================================================================
DEBUG_CONFIG = {
    "num_steps": 2,
    "num_prompts": 2,
    "group_size": 2,
    "max_tokens": 10,  # Short generation for easy debugging
    "temperature": 1.0,
    "model_name": "Qwen/Qwen3-8B-Base",
    # Sampler checkpoint for creating sampling client (uses /sampler_weights/ path)
    "sampler_checkpoint": "tinker://6eb6acdc-66a8-54d1-b01c-d6bf5731e098:train:0/sampler_weights/final",
    # Set to None to use base model:
    # "sampler_checkpoint": None,
}

# Simple synthetic math problems for testing
TEST_PROMPTS = [
    {
        "problem": "What is 2 + 3?",
        "answer": "5",
    },
    {
        "problem": "What is 7 - 4?",
        "answer": "3",
    },
]


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def wait_for_user(message: str = "Press Enter to continue..."):
    """Pause execution and wait for user input."""
    print(colored(f"\n>>> {message}", "magenta", attrs=["bold"]))
    input()


def print_header(title: str, char: str = "=", width: int = 80):
    """Print a formatted header."""
    print(colored(char * width, "cyan"))
    print(colored(f" {title} ".center(width, char), "cyan", attrs=["bold"]))
    print(colored(char * width, "cyan"))


def print_subheader(title: str):
    """Print a formatted subheader."""
    print(colored(f"\n=== {title} ===", "yellow", attrs=["bold"]))


def format_token_with_id(token_id: int, tokenizer: Tokenizer) -> str:
    """Format a token showing both ID and decoded text."""
    decoded = tokenizer.decode([token_id])
    # Escape special characters for display
    escaped = repr(decoded)[1:-1]  # Remove quotes from repr
    return f"[{token_id}] '{escaped}'"


def print_colorized_with_ids(
    tokens: list[int],
    weights: list[float],
    tokenizer: Tokenizer,
    label: str
):
    """
    Print tokens with color coding AND token IDs.
    
    Colors:
    - Yellow: weight = 0 (prompt tokens, no loss)
    - Green: weight > 0 (generated tokens, loss computed)
    """
    print_subheader(label)
    
    # First, print the color-coded string
    colorized = format_colorized(tokens, weights, tokenizer, draw_newline_arrow=True)
    print(colorized)
    
    # Then print token-by-token breakdown
    print(colored("\n--- Token-by-token breakdown ---", "white"))
    for i, (tok, w) in enumerate(zip(tokens, weights)):
        decoded = tokenizer.decode([tok])
        escaped = repr(decoded)[1:-1]
        
        if w > 0:
            color = "green"
            label_str = "GEN"
        else:
            color = "yellow"
            label_str = "PROMPT"
        
        line = f"  [{i:3d}] ID={tok:6d}  [{label_str:6s}]  '{escaped}'"
        print(colored(line, color))


def print_per_token_kl(
    student_logprobs: torch.Tensor,
    teacher_logprobs: torch.Tensor,
    mask: torch.Tensor,
    tokens: list[int],
    tokenizer: Tokenizer
):
    """Print per-token KL divergence values."""
    print_subheader("PER-TOKEN KL LOSS")
    
    kl_values = (student_logprobs - teacher_logprobs) * mask.float()
    
    total_kl = 0.0
    num_generated = 0
    
    for i in range(len(tokens)):
        decoded = tokenizer.decode([tokens[i]])
        escaped = repr(decoded)[1:-1]
        
        kl_val = kl_values[i].item() if i < len(kl_values) else 0.0
        mask_val = mask[i].item() if i < len(mask) else 0.0
        student_lp = student_logprobs[i].item() if i < len(student_logprobs) else 0.0
        teacher_lp = teacher_logprobs[i].item() if i < len(teacher_logprobs) else 0.0
        
        if mask_val > 0.5:
            color = "green"
            label = "GEN"
            total_kl += kl_val
            num_generated += 1
        else:
            color = "yellow"
            label = "PROMPT"
        
        line = (
            f"  [{i:3d}] '{escaped:15s}' | "
            f"student_lp={student_lp:8.4f} | "
            f"teacher_lp={teacher_lp:8.4f} | "
            f"KL={kl_val:8.4f} | "
            f"mask={mask_val:.0f} [{label}]"
        )
        print(colored(line, color))
    
    avg_kl = total_kl / max(num_generated, 1)
    print(colored(f"\n  Average KL over generated tokens: {avg_kl:.6f}", "cyan", attrs=["bold"]))
    print(colored(f"  Total KL sum: {total_kl:.6f}", "cyan"))
    print(colored(f"  Number of generated tokens: {num_generated}", "cyan"))


# =============================================================================
# MAIN DEBUG LOGIC
# =============================================================================

class DebugSelfDistillationDataset:
    """A simple dataset with hardcoded test prompts."""
    
    def __init__(
        self,
        prompts: list[dict],
        group_size: int,
        renderer: renderers.Renderer,
    ):
        self.prompts = prompts
        self.group_size = group_size
        self.renderer = renderer
    
    def get_batch(self, index: int) -> Sequence[EnvGroupBuilder]:
        """Return one prompt per batch."""
        if index >= len(self.prompts):
            return []
        
        prompt_data = self.prompts[index]
        return [
            ProblemGroupBuilder(
                env_thunk=partial(
                    SelfDistillationEnv,
                    prompt_data["problem"],
                    prompt_data["answer"],
                    self.renderer,
                    convo_prefix=None,
                    student_prompt_suffix=DEFAULT_STUDENT_SUFFIX,
                    proxy_teacher_template=DEFAULT_PROXY_TEACHER_TEMPLATE,
                ),
                num_envs=self.group_size,
                dataset_name="debug_test",
            )
        ]
    
    def __len__(self) -> int:
        return len(self.prompts)


async def debug_incorporate_self_distillation_kl(
    data_D: list[tinker.Datum],
    sampling_client: tinker.SamplingClient,
    trajectory_groups_P: list[TrajectoryGroup],
    env_groups_P: list[Sequence],
    metadata_D: list[dict[str, int]],
    renderer: renderers.Renderer,
    tokenizer: Tokenizer,
    kl_penalty_coef: float = 1.0,
):
    """
    Debug version of incorporate_self_distillation_kl with detailed printing.
    
    This mirrors the logic in train_self_distillation.py but adds extensive
    debug output at each step.
    """
    
    for i, datum in enumerate(data_D):
        print_header(f"DATUM {i}", "=", 80)
        
        # Get metadata
        group_idx = metadata_D[i]["group_idx"]
        traj_idx = metadata_D[i]["traj_idx"]
        
        # Get the environment
        env = env_groups_P[group_idx][traj_idx]
        if not isinstance(env, SelfDistillationEnv):
            print(colored(f"ERROR: Expected SelfDistillationEnv, got {type(env)}", "red"))
            continue
        
        # =====================================================================
        # STUDENT INPUT
        # =====================================================================
        
        # Get mask and target_tokens from datum
        mask = datum.loss_fn_inputs["mask"].to_torch()
        target_tokens = datum.loss_fn_inputs["target_tokens"].to_torch()
        student_logprobs = datum.loss_fn_inputs["logprobs"].to_torch()
        
        # Reconstruct full student sequence
        # The datum's model_input contains the input tokens (shifted by 1)
        # target_tokens is what we're predicting
        student_input_tokens = datum.model_input.to_ints()
        # Full sequence = input + last target token
        full_student_tokens = student_input_tokens + [target_tokens[-1].item()]
        
        # Create weights for color coding (0 for prompt, 1 for generated)
        # We need to shift the mask to align with full tokens
        weights = [0.0] + mask.tolist()  # Prepend 0 for first token
        
        print_colorized_with_ids(
            full_student_tokens,
            weights,
            tokenizer,
            f"STUDENT INPUT (Problem: {env.problem})"
        )
        
        wait_for_user("Review student input, then press Enter...")
        
        # =====================================================================
        # PROXY TEACHER INPUT
        # =====================================================================
        
        # Build proxy teacher prompt
        proxy_prompt = env.get_proxy_teacher_prompt()
        proxy_convo = [{"role": "user", "content": proxy_prompt}]
        proxy_input = renderer.build_generation_prompt(proxy_convo)
        
        # Extract only generated tokens (where mask=1)
        generated_mask = mask > 0.5
        generated_tokens = target_tokens[generated_mask].tolist()
        
        # Full proxy teacher sequence: concatenate proxy prompt + generated tokens
        proxy_full_tokens = proxy_input.to_ints() + generated_tokens
        proxy_full_input = tinker.ModelInput.from_ints(proxy_full_tokens)
        
        # Create weights for proxy teacher (0 for proxy prompt, 1 for copied trace)
        proxy_prompt_len = proxy_input.length
        proxy_weights = [0.0] * proxy_prompt_len + [1.0] * len(generated_tokens)
        
        print_colorized_with_ids(
            proxy_full_tokens,
            proxy_weights,
            tokenizer,
            f"PROXY TEACHER INPUT (with ground truth: {env.answer})"
        )
        
        wait_for_user("Review proxy teacher input, then press Enter...")
        
        # =====================================================================
        # COMPUTE PROXY TEACHER LOGPROBS
        # =====================================================================
        
        print_subheader("COMPUTING PROXY TEACHER LOGPROBS")
        print(colored("  Calling sampling_client.compute_logprobs_async()...", "white"))
        
        # Verify this doesn't require gradients
        print(colored("  Note: compute_logprobs_async returns values only, no gradients", "cyan"))
        
        proxy_teacher_logprobs_raw = await sampling_client.compute_logprobs_async(proxy_full_input)
        
        print(colored(f"  Received {len(proxy_teacher_logprobs_raw)} logprob values", "white"))
        
        # Extract teacher logprobs for generated tokens only
        teacher_logprobs_generated = torch.tensor(proxy_teacher_logprobs_raw[proxy_prompt_len:])
        
        # Create full-length teacher logprobs tensor
        teacher_logprobs_full = torch.zeros_like(student_logprobs)
        generated_positions = (mask > 0.5).nonzero(as_tuple=True)[0]
        
        if len(generated_positions) == len(teacher_logprobs_generated):
            teacher_logprobs_full[generated_positions] = teacher_logprobs_generated
        else:
            print(colored(
                f"  WARNING: Length mismatch! {len(generated_positions)} positions vs "
                f"{len(teacher_logprobs_generated)} teacher logprobs",
                "red"
            ))
        
        wait_for_user("Review logprob computation, then press Enter...")
        
        # =====================================================================
        # PER-TOKEN KL LOSS
        # =====================================================================
        
        # Note: We use target_tokens for display since that's what we're predicting
        print_per_token_kl(
            student_logprobs,
            teacher_logprobs_full,
            mask,
            target_tokens.tolist(),
            tokenizer
        )
        
        wait_for_user("Review per-token KL loss, then press Enter...")
        
        # =====================================================================
        # GRADIENT FLOW CHECK
        # =====================================================================
        
        print_subheader("GRADIENT FLOW VERIFICATION")
        
        # The student_logprobs come from the trajectory sampling, which uses the
        # current model weights but sampling_client doesn't track gradients
        print(colored("  Checking gradient requirements...", "white"))
        
        # teacher_logprobs_full is from compute_logprobs_async - no gradients
        print(colored(f"    teacher_logprobs requires_grad: {teacher_logprobs_full.requires_grad}", 
                      "green" if not teacher_logprobs_full.requires_grad else "red"))
        
        # student_logprobs from datum - these are stored values, not live tensors
        print(colored(f"    student_logprobs requires_grad: {student_logprobs.requires_grad}",
                      "white"))
        
        print(colored("\n  Verification summary:", "cyan", attrs=["bold"]))
        print(colored("    - Proxy teacher logprobs: ✓ No gradients (just values)", "green"))
        print(colored("    - Training update: Will recompute forward pass on student model", "cyan"))
        print(colored("    - KL advantage: Used to weight the IS/PPO loss", "cyan"))
        
        wait_for_user("Review gradient info, then press Enter...")
        
        print("\n" + "=" * 80 + "\n")


async def run_debug_training():
    """Main debug training loop."""
    
    print_header("SELF-DISTILLATION DEBUG TEST", "=", 80)
    print(colored("\nConfiguration:", "cyan", attrs=["bold"]))
    for key, value in DEBUG_CONFIG.items():
        print(f"  {key}: {value}")
    print()
    
    wait_for_user("Starting debug training. Press Enter to begin...")
    
    # Create service client
    service_client = tinker.ServiceClient()
    
    # Create sampling client directly from checkpoint or base model
    if DEBUG_CONFIG.get("sampler_checkpoint"):
        print(colored(f"\nCreating sampling client from: {DEBUG_CONFIG['sampler_checkpoint']}", "cyan"))
        sampling_client = service_client.create_sampling_client(
            model_path=DEBUG_CONFIG["sampler_checkpoint"],
            base_model=DEBUG_CONFIG["model_name"],
        )
    else:
        print(colored(f"\nCreating sampling client from base model: {DEBUG_CONFIG['model_name']}", "cyan"))
        sampling_client = service_client.create_sampling_client(
            base_model=DEBUG_CONFIG["model_name"]
        )
    
    # Get tokenizer and renderer
    tokenizer = get_tokenizer(DEBUG_CONFIG["model_name"])
    renderer = renderers.get_renderer("qwen3", tokenizer=tokenizer)
    
    # Create debug dataset
    dataset = DebugSelfDistillationDataset(
        prompts=TEST_PROMPTS[:DEBUG_CONFIG["num_prompts"]],
        group_size=DEBUG_CONFIG["group_size"],
        renderer=renderer,
    )
    
    print(colored(f"\nDataset has {len(dataset)} batches", "cyan"))
    
    # Training loop
    for step in range(DEBUG_CONFIG["num_steps"]):
        print_header(f"TRAINING STEP {step}", "#", 80)
        
        # Get batch
        env_group_builders = dataset.get_batch(step % len(dataset))
        if not env_group_builders:
            print(colored("No more batches available", "yellow"))
            break
        
        # Sample trajectories and keep envs for proxy teacher
        env_groups_P = []
        trajectory_groups_P = []
        
        print(colored("\nSampling trajectories...", "white"))
        
        for builder in env_group_builders:
            envs = await builder.make_envs()
            env_groups_P.append(envs)
            
            # Print the prompts being used
            print(colored(f"\n  Problem: {envs[0].problem}", "cyan"))
            print(colored(f"  Answer: {envs[0].answer}", "cyan"))
            print(colored(f"  Student prompt: {envs[0].get_question()}", "white"))
            print(colored(f"  Group size: {len(envs)}", "white"))
            
            trajectory_group = await do_group_rollout_and_filter_constant_reward(
                sampling_client,
                builder,
                temperature=DEBUG_CONFIG["temperature"],
                max_tokens=DEBUG_CONFIG["max_tokens"],
                do_remove_constant_reward_groups=False,
            )
            
            if trajectory_group is not None:
                trajectory_groups_P.append(trajectory_group)
        
        if not trajectory_groups_P:
            print(colored("No valid trajectories!", "red"))
            continue
        
        wait_for_user("Trajectories sampled. Press Enter to see training data...")
        
        # Assemble training data
        advantages_P = compute_advantages(trajectory_groups_P)
        data_D, metadata_D = assemble_training_data(trajectory_groups_P, advantages_P)
        
        print(colored(f"\nAssembled {len(data_D)} training datums", "cyan"))
        
        wait_for_user("Press Enter to inspect each datum in detail...")
        
        # Debug KL computation for all datums
        await debug_incorporate_self_distillation_kl(
            data_D,
            sampling_client,
            trajectory_groups_P,
            env_groups_P,
            metadata_D,
            renderer,
            tokenizer,
        )
        
        print_header(f"STEP {step} COMPLETE", "-", 80)
        print(colored("\nNote: In actual training, the KL advantages would be used", "cyan"))
        print(colored("to weight the importance sampling or PPO loss.", "cyan"))
        
        wait_for_user("Press Enter to continue to next step...")
    
    print_header("DEBUG TRAINING COMPLETE", "=", 80)
    print(colored("\nAll steps completed successfully!", "green", attrs=["bold"]))


def main():
    """Entry point."""
    asyncio.run(run_debug_training())


if __name__ == "__main__":
    main()

"""
python -m tinker_cookbook.recipes.dis
tillation.self_distillation model_name=Qwen/Qwen3-8B-Base load_checkpoint_path=tinker://
6eb6acdc-66a8-54d1-b01c-d6bf5731e098:train:0/weights/final learning_rate=1e-4 groups_per
_batch=128  group_size=8  lora_rank=128 load_optimizer_state=False  eval_aime24=True  ev
al_aime25=True  max_steps=201  wandb_project=self_distillation
"""
