"""
Standalone script to run RLMathEvaluator on a tinker checkpoint.

This script takes a checkpoint path and runs the same math evaluation that is used
as an infrequent eval in off_policy_reasoning_custom_data.py.

Example usage:
    python -m tinker_cookbook.recipes.distillation.run_rl_math_eval \
        model_path=tinker://YOUR_CHECKPOINT_PATH \
        eval_aime24=True \
        eval_aime25=True \
        temperature=0.6 \
        max_tokens=16384 \
        n_samples=4 \
        wandb_project=sft \
        wandb_run_id=<RUN_ID> \
        wandb_step=<STEP>

    # Or with explicit base model:
    python -m tinker_cookbook.recipes.distillation.run_rl_math_eval \
        model_path=tinker://YOUR_CHECKPOINT_PATH \
        model_name=Qwen/Qwen3-8B-Base \
        eval_aime24=True
"""

import asyncio
import logging
import os

import chz
import tinker
from tinker_cookbook import model_info
from tinker_cookbook.recipes.distillation.rl_math_evaluator import (
    RLMathEvaluator,
    RLMathEvaluatorBuilder,
)

logger = logging.getLogger(__name__)

# Set your API keys here or via environment variables
# os.environ['TINKER_API_KEY'] = '<your_key>'

@chz.chz
class CLIConfig:
    """Command-line configuration for standalone RL math evaluation."""

    # Model/checkpoint configuration
    model_path: str | None = None  # tinker:// path to checkpoint
    model_name: str | None = None  # Base model name (auto-detected if not provided)
    renderer_name: str | None = "qwen3"  # Renderer name

    # Dataset configuration
    eval_aime24: bool = False
    eval_aime25: bool = False
    custom_dataset: str | None = None  # Custom HuggingFace dataset name if not using AIME

    # Evaluation hyperparameters
    temperature: float = 0.6
    max_tokens: int = 16384
    n_samples: int = 4  # Number of samples per problem for pass@k calculation
    max_problems: int | None = None  # Limit number of problems to evaluate

    # W&B configuration
    wandb_project: str | None = None  # W&B project name (if not set, no logging)
    wandb_run_id: str | None = None  # Existing W&B run ID to resume (optional)
    wandb_step: int | None = None  # Step number to log at (required if logging to W&B)


async def main(config: CLIConfig):
    """Run RL math evaluation on a checkpoint."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    if config.model_path is None and config.model_name is None:
        raise ValueError("Either model_path or model_name must be provided")

    service_client = tinker.ServiceClient()

    # Auto-detect base model from checkpoint if not provided
    base_model = config.model_name
    if config.model_path is not None:
        rest_client = service_client.create_rest_client()
        training_run = await rest_client.get_training_run_by_tinker_path_async(
            config.model_path
        )
        if base_model:
            if base_model != training_run.base_model:
                raise ValueError(
                    f"Provided model_name {base_model} does not match "
                    f"checkpoint's base model {training_run.base_model}"
                )
        else:
            base_model = training_run.base_model
            logger.info(f"Auto-detected base model from checkpoint: {base_model}")

    if base_model is None:
        raise ValueError("Could not determine base model. Please provide model_name.")

    # Get renderer name
    renderer_name = config.renderer_name or model_info.get_recommended_renderer_name(
        base_model
    )
    logger.info(f"Using renderer: {renderer_name}")

    # Create sampling client
    if config.model_path:
        logger.info(f"Creating sampling client with checkpoint: {config.model_path}")
        sampling_client = service_client.create_sampling_client(
            model_path=config.model_path, base_model=base_model
        )
    else:
        logger.info(f"Creating sampling client for base model: {base_model}")
        sampling_client = service_client.create_sampling_client(base_model=base_model)

    # Build list of evaluators
    evaluators = []

    if config.eval_aime24:
        evaluators.append(
            RLMathEvaluatorBuilder(
                dataset_name="Maxwell-Jia/AIME_2024",
                split="train",
                temperature=config.temperature,
                max_tokens=config.max_tokens,
                n_samples=config.n_samples,
                max_samples=config.max_problems,
                renderer_name=renderer_name,
                model_name=base_model,
            )
        )

    if config.eval_aime25:
        evaluators.append(
            RLMathEvaluatorBuilder(
                dataset_name="math-ai/aime25",
                split="test",
                temperature=config.temperature,
                max_tokens=config.max_tokens,
                n_samples=config.n_samples,
                max_samples=config.max_problems,
                renderer_name=renderer_name,
                model_name=base_model,
            )
        )

    if config.custom_dataset:
        evaluators.append(
            RLMathEvaluatorBuilder(
                dataset_name=config.custom_dataset,
                split="test",
                temperature=config.temperature,
                max_tokens=config.max_tokens,
                n_samples=config.n_samples,
                max_samples=config.max_problems,
                renderer_name=renderer_name,
                model_name=base_model,
            )
        )

    if not evaluators:
        raise ValueError(
            "No evaluations specified. Use eval_aime24=True, eval_aime25=True, "
            "or custom_dataset=<dataset_name>"
        )

    # Run evaluations
    all_metrics = {}
    for builder in evaluators:
        logger.info(f"Running evaluation on {builder.dataset_name}...")
        evaluator = builder()
        metrics = await evaluator(sampling_client)
        all_metrics.update(metrics)

    # Print summary
    print("\n" + "=" * 80)
    print("EVALUATION RESULTS")
    print("=" * 80)
    if config.model_path:
        print(f"Checkpoint: {config.model_path}")
    print(f"Base Model: {base_model}")
    print(f"Temperature: {config.temperature}")
    print(f"N Samples: {config.n_samples}")
    print("-" * 80)

    for metric_name, metric_value in sorted(all_metrics.items()):
        if isinstance(metric_value, float):
            print(f"  {metric_name}: {metric_value:.4f}")
        else:
            print(f"  {metric_name}: {metric_value}")

    print("=" * 80)

    # Log to W&B if configured
    if config.wandb_project:
        import wandb

        if config.wandb_step is None:
            logger.warning(
                "wandb_step not provided. Logging without step. "
                "Consider providing wandb_step for proper x-axis alignment."
            )

        if config.wandb_run_id:
            # Resume existing run
            logger.info(f"Resuming W&B run: {config.wandb_run_id}")
            wandb.init(
                project=config.wandb_project,
                id=config.wandb_run_id,
                resume="must",
            )
        else:
            # Create new run
            run_name = config.model_path or base_model
            logger.info(f"Creating new W&B run: {run_name}")
            wandb.init(
                project=config.wandb_project,
                name=run_name,
                config={
                    "model_path": config.model_path,
                    "base_model": base_model,
                    "temperature": config.temperature,
                    "n_samples": config.n_samples,
                    "max_tokens": config.max_tokens,
                },
            )

        # Log metrics
        if config.wandb_step is not None:
            wandb.log(all_metrics, step=config.wandb_step)
            logger.info(f"Logged metrics to W&B at step {config.wandb_step}")
        else:
            wandb.log(all_metrics)
            logger.info("Logged metrics to W&B")

        wandb.finish()

    return all_metrics


if __name__ == "__main__":
    config = chz.entrypoint(CLIConfig)
    asyncio.run(main(config))
