"""
Implements feedback-based on-policy self-distillation training.

The key difference from standard self-distillation:
1. Sample G rollouts per prompt from the student
2. Extract summaries from rollouts (content after </think>)
3. Generate textual feedback f based on summaries + ground truth
4. Create proxy teacher conditioned on the generated feedback
5. Train using KL divergence between student and proxy teacher
"""

import asyncio
import logging
import os
import time
from typing import Any, Dict, List, Sequence, cast

import chz
import tinker
import torch

from tinker.types import LossFnType

from tinker_cookbook import checkpoint_utils
from tinker_cookbook.display import colorize_example
from tinker_cookbook.distillation.feedback_self_distillation_datasets import (
    FeedbackSelfDistillationDataset,
    FeedbackSelfDistillationDatasetBuilder,
    FeedbackSelfDistillationEnv,
    extract_summary_from_response,
)
from tinker_cookbook.eval.evaluators import SamplingClientEvaluator, SamplingClientEvaluatorBuilder
from tinker_cookbook.rl.data_processing import (
    assemble_training_data,
    compute_advantages,
)
from tinker_cookbook.rl.metric_util import compute_trajectory_metrics
from tinker_cookbook.rl.metrics import discounted_future_sum_vectorized
from tinker_cookbook.rl.train import (
    compute_full_batch_metrics_and_get_sampling_client,
    do_group_rollout_and_filter_constant_reward,
    save_checkpoint_and_get_sampling_client,
    train_step,
)
from tinker_cookbook.rl.types import (
    EnvGroupBuilder,
    TrajectoryGroup,
)
from tinker_cookbook.tokenizer_utils import Tokenizer
from tinker_cookbook.utils import ml_log
from tinker_cookbook.utils.misc_utils import safezip, timed
from tinker_cookbook.utils.trace import scope, update_scope_context, trace_init
from tinker_cookbook import renderers

logger = logging.getLogger(__name__)


@scope
async def generate_feedback_for_group(
    sampling_client: tinker.SamplingClient,
    renderer: renderers.Renderer,
    envs: Sequence[FeedbackSelfDistillationEnv],
    trajectory_group: TrajectoryGroup,
    tokenizer: Tokenizer,
    filter_incomplete_traces: bool,
    feedback_max_tokens: int,
    feedback_temperature: float,
) -> str:
    """
    Generate feedback for a group of rollouts.
    
    1. Extract summaries from each rollout (content after </think>)
    2. Format summaries into the feedback prompt
    3. Generate feedback text using the sampling client
    
    Args:
        sampling_client: Client for generating feedback
        renderer: For building generation prompts
        envs: The environments (all share same problem/answer)
        trajectory_group: Contains the rollouts from student
        tokenizer: For decoding responses
        filter_incomplete_traces: If True, skip traces without </think>
        feedback_max_tokens: Max tokens for feedback generation
        feedback_temperature: Temperature for feedback generation
        
    Returns:
        Generated feedback text
    """
    # Extract summaries from each trajectory
    summaries = []
    for i, trajectory in enumerate(trajectory_group.trajectories):
        # Get the full response text
        if trajectory.transitions:
            action_tokens = trajectory.transitions[0].ac.tokens
            response_text = tokenizer.decode(action_tokens)
            
            # Extract summary (content after </think>)
            summary = extract_summary_from_response(response_text, filter_incomplete_traces)
            if summary is not None:
                summaries.append(f"Student solution {i+1}: {summary}")
    
    # If no valid summaries, use a fallback
    if not summaries:
        summaries_text = "(No valid summaries available - students did not complete their responses)"
    else:
        summaries_text = "\n".join(summaries)
    
    # Get feedback prompt from the first env (all share same problem/answer)
    feedback_prompt = envs[0].get_feedback_prompt(summaries_text)
    
    # Build generation prompt for feedback
    feedback_convo = (envs[0].convo_prefix or []) + [
        {"role": "user", "content": feedback_prompt},
    ]
    feedback_input = renderer.build_generation_prompt(feedback_convo)
    
    # Generate feedback
    feedback_response = await sampling_client.sample_async(
        feedback_input,
        max_tokens=feedback_max_tokens,
        temperature=feedback_temperature,
        stop_sequences=renderer.get_stop_sequences(),
    )
    
    # Decode feedback
    feedback_text = tokenizer.decode(feedback_response.tokens)
    
    # Parse to remove stop tokens if present
    feedback_message, _ = renderer.parse_response(feedback_response.tokens)
    feedback_text = feedback_message["content"]
    feedback_text = extract_summary_from_response(feedback_text, filter_incomplete_traces)
    
    return feedback_text


@scope
async def incorporate_feedback_self_distillation_kl(
    data_D: List[tinker.Datum],
    sampling_client: tinker.SamplingClient,
    trajectory_groups_P: list[TrajectoryGroup],
    env_groups_P: list[Sequence[FeedbackSelfDistillationEnv]],
    metadata_D: List[dict[str, int]],
    renderer: renderers.Renderer,
    kl_penalty_coef: float,
    kl_discount_factor: float,
) -> Dict[str, float]:
    """
    Compute reverse KL between student and proxy teacher (conditioned on generated feedback).
    
    For each datum:
    - Student: generated the trace with student prompt
    - Proxy Teacher: same model computing logprobs on the same trace, but with 
      a feedback-conditioned prompt prefix
    
    The advantage is set to -kl_penalty_coef * reverse_KL.
    """
    
    # Build proxy teacher inputs for each datum
    proxy_teacher_sequences = []
    generated_token_counts = []  # Track how many generated tokens per datum
    
    for i, datum in enumerate(data_D):
        # Get the metadata to find which trajectory/env this datum came from
        group_idx = metadata_D[i]["group_idx"]
        traj_idx = metadata_D[i]["traj_idx"]
        
        # Get the environment (which now has generated_feedback set)
        env = env_groups_P[group_idx][traj_idx]
        
        if not isinstance(env, FeedbackSelfDistillationEnv):
            raise TypeError(f"Expected FeedbackSelfDistillationEnv, got {type(env)}")
        
        # Build the proxy teacher prompt input (conditioned on feedback)
        proxy_prompt = env.get_proxy_teacher_prompt()
        proxy_convo = (env.convo_prefix or []) + [
            {"role": "user", "content": proxy_prompt},
        ]
        proxy_input = renderer.build_generation_prompt(proxy_convo)
        
        # Get the mask and target_tokens
        mask = datum.loss_fn_inputs["mask"].to_torch()
        target_tokens = datum.loss_fn_inputs["target_tokens"].to_torch()
        
        # Extract ONLY the generated tokens (where mask=1)
        generated_mask = mask > 0.5  # Boolean mask
        generated_tokens = target_tokens[generated_mask].tolist()
        generated_token_counts.append(len(generated_tokens))
        
        # Construct full sequence: proxy_prefix + generated_tokens_only
        full_sequence = tinker.ModelInput.from_ints(proxy_input.to_ints() + generated_tokens)
        proxy_teacher_sequences.append(full_sequence)
    
    # Compute proxy teacher logprobs for all sequences
    proxy_teacher_logprobs_D = await asyncio.gather(
        *[
            sampling_client.compute_logprobs_async(seq)
            for seq in proxy_teacher_sequences
        ]
    )
    
    # Compute reverse KL
    sampled_logprobs_D = [datum.loss_fn_inputs["logprobs"].to_torch() for datum in data_D]
    float_masks = [datum.loss_fn_inputs["mask"].to_torch().float() for datum in data_D]
    
    reverse_kl = []
    for i, datum in enumerate(data_D):
        mask = float_masks[i]
        student_logprobs = sampled_logprobs_D[i]
        
        # Get teacher logprobs for the generated tokens only
        proxy_prefix_len = proxy_teacher_sequences[i].length - generated_token_counts[i]
        teacher_logprobs_generated = torch.tensor(proxy_teacher_logprobs_D[i][proxy_prefix_len:])
        
        # Create a full-length tensor with 0s for prompt positions
        teacher_logprobs_full = torch.zeros_like(student_logprobs)
        
        # Find the positions where mask=1 and fill in teacher logprobs
        generated_positions = (mask > 0.5).nonzero(as_tuple=True)[0]
        if len(generated_positions) == len(teacher_logprobs_generated):
            teacher_logprobs_full[generated_positions] = teacher_logprobs_generated
        else:
            logger.warning(
                f"Datum {i}: Length mismatch - {len(generated_positions)} generated positions "
                f"vs {len(teacher_logprobs_generated)} teacher logprobs. Skipping KL for this datum."
            )
        
        # Compute reverse KL: KL[p||q] = log p - log q
        kl = (student_logprobs - teacher_logprobs_full) * mask
        reverse_kl.append(kl)
    
    # Update advantages with negative KL
    for i, datum in enumerate(data_D):
        kl_advantages = -kl_penalty_coef * reverse_kl[i]
        if kl_discount_factor > 0:
            kl_advantages = torch.tensor(
                discounted_future_sum_vectorized(kl_advantages.numpy(), kl_discount_factor)
            )
        # For self-distillation, KL IS the advantage (no reward component)
        datum.loss_fn_inputs["advantages"] = tinker.TensorData.from_torch(kl_advantages)
    
    # Compute average reverse KL for logging
    total_kl = sum([kl.sum().item() for kl in reverse_kl])
    total_mask = sum([mask.sum().item() for mask in float_masks])
    avg_reverse_kl = total_kl / max(total_mask, 1.0)
    
    return {"feedback_self_distill_kl": float(avg_reverse_kl)}


@chz.chz
class Config:
    learning_rate: float
    dataset_builder: FeedbackSelfDistillationDatasetBuilder
    model_name: str
    max_tokens: int
    temperature: float = 1.0
    evaluator_builders: list[SamplingClientEvaluatorBuilder] = chz.field(default_factory=list)
    lora_rank: int = 32

    kl_penalty_coef: float = 1.0
    kl_discount_factor: float = 0.0

    # Feedback generation parameters
    filter_incomplete_traces: bool = True
    feedback_max_tokens: int = 2048
    feedback_temperature: float = 0.7
    feedback_model_base_url: str | None = None  # If None, use same model

    # Loss function to use for training: "importance_sampling" or "ppo"
    loss_fn: LossFnType = "importance_sampling"

    # Number of optimizer steps per training iteration.
    num_substeps: int = 1

    wandb_project: str | None = None
    wandb_name: str | None = None

    log_path: str = chz.field(munger=lambda _, s: os.path.expanduser(s))
    base_url: str | None = None
    enable_trace: bool = False

    eval_every: int = 50
    save_every: int = 50
    infrequent_eval_every: int = 50
    infrequent_evaluator_builders: list[SamplingClientEvaluatorBuilder] = chz.field(default_factory=list)
    load_checkpoint_path: str | None = None
    load_optimizer_state: bool = True  # If False, only load weights (not optimizer state)
    max_steps: int | None = None  # If None, train on full dataset


@scope
async def prepare_minibatch(
    env_group_builders_P: Sequence[EnvGroupBuilder],
    trajectory_groups_P: list[TrajectoryGroup],
    env_groups_P: list[Sequence[FeedbackSelfDistillationEnv]],
    sampling_client: tinker.SamplingClient,
    tokenizer: Tokenizer,
    renderer: renderers.Renderer,
    kl_penalty_coef: float,
    kl_discount_factor: float,
) -> tuple[list[tinker.Datum], dict[str, Any]]:
    """Converts the trajectories into a minibatch with feedback-based KL as advantage."""

    metrics = {}
    taglist_P = [env_group_builder.logging_tags() for env_group_builder in env_group_builders_P]
    metrics.update(compute_trajectory_metrics(trajectory_groups_P, taglist_P))

    # Assemble training data
    with timed("assemble_training_data", metrics):
        advantages_P = compute_advantages(trajectory_groups_P)
        data_D, metadata_D = assemble_training_data(trajectory_groups_P, advantages_P)

    # Print one example for debugging
    if data_D:
        logger.info(colorize_example(data_D[0], tokenizer, key="mask"))

    # Incorporate feedback-based self-distillation KL as the advantage
    with timed("compute_feedback_kl", metrics):
        kl_metrics = await incorporate_feedback_self_distillation_kl(
            data_D,
            sampling_client,
            trajectory_groups_P,
            env_groups_P,
            metadata_D,
            renderer,
            kl_penalty_coef,
            kl_discount_factor,
        )
    metrics.update(kl_metrics)

    return data_D, metrics


@scope
async def do_train_step_and_get_sampling_client(
    cfg: Config,
    i_batch: int,
    training_client: tinker.TrainingClient,
    sampling_client: tinker.SamplingClient,
    tokenizer: Tokenizer,
    renderer: renderers.Renderer,
    env_group_builders_P: Sequence[EnvGroupBuilder],
    trajectory_groups_P: list[TrajectoryGroup],
    env_groups_P: list[Sequence[FeedbackSelfDistillationEnv]],
) -> tuple[tinker.SamplingClient, dict[str, Any]]:
    update_scope_context({"step": i_batch})

    metrics = {}
    data_D, prepare_minibatch_metrics = await prepare_minibatch(
        env_group_builders_P,
        trajectory_groups_P,
        env_groups_P,
        sampling_client,
        tokenizer,
        renderer,
        kl_penalty_coef=cfg.kl_penalty_coef,
        kl_discount_factor=cfg.kl_discount_factor,
    )
    metrics.update(prepare_minibatch_metrics)

    with timed("train", metrics):
        training_logprobs_D = await train_step(
            data_D,
            training_client,
            cfg.learning_rate,
            cfg.num_substeps,
            cfg.loss_fn,
        )

    new_sampling_client, full_batch_metrics = await compute_full_batch_metrics_and_get_sampling_client(
        training_client,
        i_batch + 1,
        data_D,
        training_logprobs_D,
        cfg.log_path,
        cfg.save_every,
        compute_post_kl=False,
    )
    metrics.update(full_batch_metrics)

    return new_sampling_client, metrics


@scope
async def do_sync_training(
    start_batch: int,
    end_batch: int,
    num_batches: int,
    cfg: Config,
    training_client: tinker.TrainingClient,
    evaluators: list[SamplingClientEvaluator],
    infrequent_evaluators: list[SamplingClientEvaluator],
    dataset: FeedbackSelfDistillationDataset,
    ml_logger: ml_log.Logger,
    tokenizer: Tokenizer,
    renderer: renderers.Renderer,
    feedback_sampling_client: tinker.SamplingClient | None = None,
):
    """Implements fully synchronous feedback-based self-distillation training."""

    # Initial sampling client
    sampling_client, _ = await save_checkpoint_and_get_sampling_client(
        training_client, start_batch, cfg.log_path, cfg.save_every
    )
    
    # Use separate feedback client if provided, otherwise use same model
    if feedback_sampling_client is None:
        feedback_sampling_client = sampling_client

    for i_batch in range(start_batch, end_batch):
        metrics = {
            "progress/batch": i_batch,
            "optim/lr": cfg.learning_rate,
            "progress/done_frac": (i_batch + 1) / num_batches,
        }
        t_start = time.time()

        # Run evaluations
        if cfg.eval_every > 0 and i_batch % cfg.eval_every == 0:
            with timed("run_evals", metrics):
                for evaluator in evaluators:
                    eval_metrics = await evaluator(sampling_client)
                    metrics.update({f"test/{k}": v for k, v in eval_metrics.items()})

        # Run infrequent evaluations (e.g., AIME24, AIME25)
        if cfg.infrequent_eval_every > 0 and i_batch % cfg.infrequent_eval_every == 0:
            with timed("run_infrequent_evals", metrics):
                for evaluator in infrequent_evaluators:
                    eval_metrics = await evaluator(sampling_client)
                    metrics.update({f"test/{k}": v for k, v in eval_metrics.items()})

        # Get batch
        env_group_builders_P = dataset.get_batch(i_batch)
        
        # Phase 1: Sample trajectories AND keep track of envs
        env_groups_P: list[Sequence[FeedbackSelfDistillationEnv]] = []
        trajectory_groups_P: list[TrajectoryGroup] = []
        valid_env_group_builders_P: list[EnvGroupBuilder] = []
        
        with timed("sample", metrics):
            for builder in env_group_builders_P:
                # Create envs and sample trajectories
                envs = await builder.make_envs()
                envs = cast(Sequence[FeedbackSelfDistillationEnv], envs)
                
                trajectory_group = await do_group_rollout_and_filter_constant_reward(
                    sampling_client,
                    builder,
                    temperature=cfg.temperature,
                    max_tokens=cfg.max_tokens,
                    do_remove_constant_reward_groups=False,
                )
                if trajectory_group is not None:
                    env_groups_P.append(envs)
                    trajectory_groups_P.append(trajectory_group)
                    valid_env_group_builders_P.append(builder)

        if not trajectory_groups_P:
            logger.warning(f"No valid trajectory groups for batch {i_batch}")
            continue

        # Phase 2: Generate feedback for each group
        with timed("generate_feedback", metrics):
            feedback_tasks = []
            for envs, trajectory_group in zip(env_groups_P, trajectory_groups_P, strict=True):
                task = generate_feedback_for_group(
                    feedback_sampling_client,
                    renderer,
                    envs,
                    trajectory_group,
                    tokenizer,
                    cfg.filter_incomplete_traces,
                    cfg.feedback_max_tokens,
                    cfg.feedback_temperature,
                )
                feedback_tasks.append(task)
            
            # Generate all feedbacks in parallel
            feedbacks = await asyncio.gather(*feedback_tasks)
            
            # Set feedback on all envs in each group
            for envs, feedback in zip(env_groups_P, feedbacks, strict=True):
                for env in envs:
                    env.generated_feedback = feedback
        
        # Log feedback for first group
        if feedbacks:
            logger.info(f"Generated feedback (first group): {feedbacks[0]}")
            metrics["feedback/length"] = sum(len(f) for f in feedbacks) / len(feedbacks)

        # Phase 3: Train step (now envs have feedback set)
        sampling_client, train_step_metrics = await do_train_step_and_get_sampling_client(
            cfg,
            i_batch,
            training_client,
            sampling_client,
            tokenizer,
            renderer,
            valid_env_group_builders_P,
            trajectory_groups_P,
            env_groups_P,
        )
        
        # Update feedback sampling client if using same model
        if cfg.feedback_model_base_url is None:
            feedback_sampling_client = sampling_client

        # Log metrics
        metrics.update(train_step_metrics)
        metrics["time/total"] = time.time() - t_start
        ml_logger.log_metrics(metrics, step=i_batch)


@scope
async def main(cfg: Config):
    """Main training loop for feedback-based self-distillation."""
    ml_logger = ml_log.setup_logging(
        log_dir=cfg.log_path,
        wandb_project=cfg.wandb_project,
        config=cfg,
        wandb_name=cfg.wandb_name,
    )
    if cfg.enable_trace:
        current_task = asyncio.current_task()
        if current_task is not None:
            current_task.set_name("main")
        trace_events_path = os.path.join(cfg.log_path, "trace_events.jsonl")
        logger.info(f"Tracing is enabled. Trace events will be saved to {trace_events_path}")
        trace_init(output_file=trace_events_path)

    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("pylatexenc").setLevel(logging.WARNING)

    resume_info = checkpoint_utils.get_last_checkpoint(cfg.log_path)
    if resume_info:
        start_batch = resume_info["batch"]
    else:
        start_batch = 0

    service_client = tinker.ServiceClient(base_url=cfg.base_url)
    training_client = await service_client.create_lora_training_client_async(
        cfg.model_name, rank=cfg.lora_rank
    )

    load_state_path: str | None = (
        resume_info["state_path"] if resume_info else cfg.load_checkpoint_path
    )
    if load_state_path:
        if cfg.load_optimizer_state:
            future = await training_client.load_state_with_optimizer_async(load_state_path)
            logger.info(f"Loading state WITH optimizer from {load_state_path}")
        else:
            future = await training_client.load_state_async(load_state_path)
            logger.info(f"Loading state WITHOUT optimizer from {load_state_path}")
        _ = await future.result_async()
        logger.info(f"Loaded state from {load_state_path}")

    # Get tokenizer and renderer
    tokenizer = training_client.get_tokenizer()
    renderer = renderers.get_renderer(cfg.dataset_builder.renderer_name, tokenizer=tokenizer)

    # Create dataset
    dataset, _ = await cfg.dataset_builder()
    num_batches = len(dataset)
    logger.info(f"Will train on {num_batches} batches")

    # Create evaluators
    evaluators = [evaluator() for evaluator in cfg.evaluator_builders]
    infrequent_evaluators = [evaluator() for evaluator in cfg.infrequent_evaluator_builders]

    # Create feedback sampling client if using separate model
    feedback_sampling_client = None
    if cfg.feedback_model_base_url:
        feedback_service_client = tinker.ServiceClient(base_url=cfg.feedback_model_base_url)
        feedback_sampling_client = await feedback_service_client.create_sampling_client_async(
            cfg.model_name
        )
        logger.info(f"Using separate feedback model at {cfg.feedback_model_base_url}")

    # Compute end_batch (respect max_steps if set)
    end_batch = num_batches
    if cfg.max_steps is not None:
        end_batch = min(num_batches, cfg.max_steps)
        logger.info(f"max_steps={cfg.max_steps}, will train for {end_batch - start_batch} steps")

    # Training loop
    await do_sync_training(
        start_batch=start_batch,
        end_batch=end_batch,
        num_batches=num_batches,
        cfg=cfg,
        training_client=training_client,
        evaluators=evaluators,
        infrequent_evaluators=infrequent_evaluators,
        dataset=dataset,
        ml_logger=ml_logger,
        tokenizer=tokenizer,
        renderer=renderer,
        feedback_sampling_client=feedback_sampling_client,
    )

    # Save final checkpoint
    if start_batch < num_batches:
        _ = await checkpoint_utils.save_checkpoint_async(
            training_client=training_client,
            name="final",
            log_path=cfg.log_path,
            kind="both",
            loop_state={"batch": num_batches},
        )
    else:
        logger.info("Training was already complete; nothing to do")

    # Cleanup
    ml_logger.close()
    logger.info("Training completed successfully")
