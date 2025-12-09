"""
On-policy self-distillation for reasoning tasks.

This script implements self-distillation where the "teacher" is the same model
conditioned on a ground-truth-enhanced prompt. The KL divergence between
student and proxy teacher provides the training signal.

Example usage:
    python -m tinker_cookbook.recipes.distillation.self_distillation \\
        model_name=Qwen/Qwen3-8B-Base \\
        load_checkpoint_path=/path/to/sft/checkpoint \\
        learning_rate=1e-5 \\
        groups_per_batch=256 \\
        lora_rank=128 \\
        wandb_project=self_distillation
"""

import asyncio
import logging
import os
from datetime import datetime

import chz
from tinker_cookbook import cli_utils, model_info
from tinker_cookbook.distillation import train_self_distillation
from tinker_cookbook.distillation.self_distillation_datasets import (
    SelfDistillationDatasetBuilder,
    DEFAULT_STUDENT_SUFFIX,
    DEFAULT_PROXY_TEACHER_TEMPLATE,
)

logger = logging.getLogger(__name__)


@chz.chz
class CLIConfig:
    """Command-line configuration for self-distillation."""

    # Model configuration
    model_name: str = "Qwen/Qwen3-8B-Base"  # Student model
    lora_rank: int = 128
    renderer_name: str | None = None
    load_checkpoint_path: str | None = None  # Starting checkpoint (e.g., from SFT)

    # Dataset configuration
    group_size: int = 4  # Number of rollouts per prompt
    groups_per_batch: int = 256
    seed: int = 0

    # Prompt customization
    student_prompt_suffix: str = DEFAULT_STUDENT_SUFFIX
    proxy_teacher_template: str = DEFAULT_PROXY_TEACHER_TEMPLATE

    # Training hyperparameters
    learning_rate: float = 1e-5
    max_tokens: int = 4096
    temperature: float = 1.0
    kl_penalty_coef: float = 1.0
    kl_discount_factor: float = 0.0

    # Optimizer configuration
    num_substeps: int = 1
    loss_fn: str = "importance_sampling"

    # Logging configuration
    log_path: str | None = None
    wandb_project: str | None = None
    wandb_name: str | None = None

    # Evaluation and checkpointing
    eval_every: int = 20
    save_every: int = 20

    # Service configuration
    base_url: str | None = None

    behavior_if_log_dir_exists: cli_utils.LogdirBehavior = "ask"


async def cli_main(cli_config: CLIConfig):
    """Convert CLI config to full config and run training."""

    # Get renderer name
    renderer_name = cli_config.renderer_name or model_info.get_recommended_renderer_name(
        cli_config.model_name
    )

    # Create log path if not specified
    if cli_config.log_path is not None:
        log_path = cli_config.log_path
    else:
        model_name = cli_config.model_name.replace("/", "-")
        run_name = (
            f"self-distill-polaris-{model_name}-"
            f"{cli_config.lora_rank}rank-{cli_config.learning_rate}lr-"
            f"{cli_config.groups_per_batch}batch-{datetime.now().strftime('%Y-%m-%d-%H-%M')}"
        )
        log_path = os.path.expanduser(f"~/tinker-examples/self-distillation/{run_name}")

    # Create wandb name if not specified
    if cli_config.wandb_name is not None:
        wandb_name = cli_config.wandb_name
    else:
        wandb_name = os.path.basename(log_path)

    # Create dataset builder
    dataset_builder = SelfDistillationDatasetBuilder(
        groups_per_batch=cli_config.groups_per_batch,
        group_size=cli_config.group_size,
        model_name_for_tokenizer=cli_config.model_name,
        renderer_name=renderer_name,
        student_prompt_suffix=cli_config.student_prompt_suffix,
        proxy_teacher_template=cli_config.proxy_teacher_template,
        seed=cli_config.seed,
    )

    # Create full config
    config = train_self_distillation.Config(
        learning_rate=cli_config.learning_rate,
        dataset_builder=dataset_builder,
        model_name=cli_config.model_name,
        lora_rank=cli_config.lora_rank,
        max_tokens=cli_config.max_tokens,
        temperature=cli_config.temperature,
        kl_penalty_coef=cli_config.kl_penalty_coef,
        kl_discount_factor=cli_config.kl_discount_factor,
        num_substeps=cli_config.num_substeps,
        loss_fn=cli_config.loss_fn,  # type: ignore
        wandb_project=cli_config.wandb_project,
        wandb_name=wandb_name,
        log_path=log_path,
        base_url=cli_config.base_url,
        load_checkpoint_path=cli_config.load_checkpoint_path,
        eval_every=cli_config.eval_every,
        save_every=cli_config.save_every,
    )

    cli_utils.check_log_dir(log_path, behavior_if_exists=cli_config.behavior_if_log_dir_exists)

    # Run training
    await train_self_distillation.main(config)


if __name__ == "__main__":
    cli_config = chz.entrypoint(CLIConfig)
    asyncio.run(cli_main(cli_config))
