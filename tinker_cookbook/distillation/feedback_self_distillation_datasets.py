"""
Dataset utilities for feedback-based on-policy self-distillation.

This module contains dataset configuration classes and environment definitions
for self-distillation where:
1. The student generates G rollouts per prompt
2. A feedback model generates textual feedback based on rollout summaries + ground truth
3. The proxy teacher is the same model conditioned on the generated feedback
"""

import math
from functools import partial
from typing import Sequence

import chz
import tinker
from datasets import load_dataset
from tinker_cookbook import renderers
from tinker_cookbook.rl.problem_env import ProblemEnv, ProblemGroupBuilder, logger
from tinker_cookbook.rl.types import (
    Action,
    EnvGroupBuilder,
    RLDataset,
    RLDatasetBuilder,
    StepResult,
)
from tinker_cookbook.tokenizer_utils import get_tokenizer
from tinker_cookbook.recipes.math_rl.math_grading import extract_boxed


# Default prompt templates
DEFAULT_STUDENT_SUFFIX = " Write your answer in \\boxed{} format."

DEFAULT_FEEDBACK_PROMPT_TEMPLATE = """You are analyzing student attempts at solving a math problem.

Problem: {problem}
Ground Truth Answer: {answer}

Student Summaries (their final answers after thinking):
{summaries}

Based on these attempts and the ground truth, provide feedback that:
1. Identifies common mistakes, misconceptions, and pitfalls
2. Highlights the correct approach
3. Explains key steps needed to reach the solution

Do not leak the final answer in your feedback, just the feedback to guide the student on how to approach the solution. Think through, and then provide a summarized feedback now:
"""

DEFAULT_PROXY_TEACHER_TEMPLATE = """You are solving a math problem. You have received the following feedback from reviewing multiple solution attempts:

Feedback: {feedback}

Now solve the following problem step by step:
{problem}

Write your answer in \\boxed{{}} format.
"""


def extract_summary_from_response(response_text: str, filter_incomplete: bool = True) -> str | None:
    """
    Extract the summary part from a model response (after </think> tag).
    
    Args:
        response_text: Full response text from the model
        filter_incomplete: If True, return None for traces without </think> tag
        
    Returns:
        The summary text after </think>, or None if incomplete and filtering enabled
    """
    # Check for </think> tag (used by Qwen3 and similar models)
    if "</think>" in response_text:
        parts = response_text.split("</think>", 1)
        if len(parts) == 2:
            summary = parts[1].strip()
            return summary if summary else None
    
    # No </think> found - trace may be incomplete
    if filter_incomplete:
        return None
    
    # If not filtering, return the whole response as-is
    return response_text.strip() if response_text.strip() else None


class FeedbackSelfDistillationEnv(ProblemEnv):
    """
    Environment for feedback-based self-distillation that stores:
    - student prompt (just the problem)
    - generated feedback text (set after feedback generation)
    - proxy teacher prompt (problem + generated feedback)
    
    The environment returns zero reward since training signal comes from KL only.
    """

    def __init__(
        self,
        problem: str,
        answer: str,
        renderer: renderers.Renderer,
        convo_prefix: list[renderers.Message] | None = None,
        student_prompt_suffix: str = DEFAULT_STUDENT_SUFFIX,
        feedback_prompt_template: str = DEFAULT_FEEDBACK_PROMPT_TEMPLATE,
        proxy_teacher_template: str = DEFAULT_PROXY_TEACHER_TEMPLATE,
    ):
        # Set format_coef to 0 since we don't use format rewards
        super().__init__(renderer, convo_prefix, format_coef=0.0)
        self.problem = problem
        self.answer = answer
        self.student_prompt_suffix = student_prompt_suffix
        self.feedback_prompt_template = feedback_prompt_template
        self.proxy_teacher_template = proxy_teacher_template
        
        # These will be set after rollouts and feedback generation
        self.rollout_summary: str | None = None  # Summary from this env's rollout
        self.generated_feedback: str | None = None  # Feedback text (shared across group)

    def get_question(self) -> str:
        """Returns the student prompt (just the problem + suffix)."""
        return self.problem + self.student_prompt_suffix

    def get_feedback_prompt(self, summaries_text: str) -> str:
        """Returns the feedback prompt with summaries filled in."""
        return self.feedback_prompt_template.format(
            problem=self.problem,
            answer=self.answer,
            summaries=summaries_text,
        )

    def get_proxy_teacher_prompt(self) -> str:
        """
        Returns the proxy teacher prompt conditioned on generated feedback.
        
        Must be called after generated_feedback is set.
        """
        if self.generated_feedback is None:
            raise ValueError("generated_feedback must be set before calling get_proxy_teacher_prompt")
        return self.proxy_teacher_template.format(
            problem=self.problem,
            feedback=self.generated_feedback,
        )

    def check_format(self, sample_str: str) -> bool:
        """Check if sample contains \\boxed{} format."""
        try:
            _ = extract_boxed(sample_str)
            return True
        except ValueError:
            return False

    def check_answer(self, sample_str: str) -> bool:
        """Not used for self-distillation - always return False."""
        return False

    def get_reference_answer(self) -> str:
        """Return the reference answer for logging purposes."""
        return self.answer

    async def step(self, action: Action) -> StepResult:
        """Return zero reward always - training signal comes from KL only."""
        message, parse_success = self.renderer.parse_response(action)
        return StepResult(
            reward=0.0,
            episode_done=True,
            next_observation=tinker.ModelInput.empty(),
            next_stop_condition=self.stop_condition,
            metrics={},
        )


class FeedbackSelfDistillationDataset(RLDataset):
    """Dataset for feedback-based self-distillation using Polaris math problems."""

    def __init__(
        self,
        batch_size: int,
        group_size: int,
        renderer: renderers.Renderer,
        tokenizer,
        convo_prefix: list[renderers.Message] | None = None,
        student_prompt_suffix: str = DEFAULT_STUDENT_SUFFIX,
        feedback_prompt_template: str = DEFAULT_FEEDBACK_PROMPT_TEMPLATE,
        proxy_teacher_template: str = DEFAULT_PROXY_TEACHER_TEMPLATE,
        seed: int = 0,
        dataset_name: str = "polaris_feedback_selfdistill",
    ):
        self.ds = load_dataset("POLARIS-Project/Polaris-Dataset-53K", split="train").shuffle(
            seed=seed
        )
        self.batch_size = batch_size
        self.group_size = group_size
        self.renderer = renderer
        self.tokenizer = tokenizer
        self.convo_prefix = convo_prefix
        self.student_prompt_suffix = student_prompt_suffix
        self.feedback_prompt_template = feedback_prompt_template
        self.proxy_teacher_template = proxy_teacher_template
        self.dataset_name = dataset_name

    def get_batch(self, index: int) -> Sequence[EnvGroupBuilder]:
        batch_start = index * self.batch_size
        batch_end = min((index + 1) * self.batch_size, len(self.ds))
        assert batch_start < batch_end, "Incorrect batch size"
        return [
            builder
            for row in self.ds.select(range(batch_start, batch_end))
            if (builder := self._make_env_group_builder(row, self.group_size)) is not None
        ]

    def __len__(self) -> int:
        return math.ceil(len(self.ds) / self.batch_size)

    def _make_env_group_builder(
        self, x: dict[str, str], group_size: int
    ) -> ProblemGroupBuilder | None:
        # Extract problem and answer from the Polaris dataset
        problem = x.get("problem", "")
        answer = x.get("answer", "")
        if not (problem and answer):
            return None
        return ProblemGroupBuilder(
            env_thunk=partial(
                FeedbackSelfDistillationEnv,
                problem,
                answer,
                self.renderer,
                convo_prefix=self.convo_prefix,
                student_prompt_suffix=self.student_prompt_suffix,
                feedback_prompt_template=self.feedback_prompt_template,
                proxy_teacher_template=self.proxy_teacher_template,
            ),
            num_envs=group_size,
            dataset_name=self.dataset_name,
        )


@chz.chz
class FeedbackSelfDistillationDatasetBuilder(RLDatasetBuilder):
    """Builder for feedback-based self-distillation dataset."""

    groups_per_batch: int
    group_size: int
    model_name_for_tokenizer: str
    renderer_name: str
    convo_prefix: list[renderers.Message] | None = None
    student_prompt_suffix: str = DEFAULT_STUDENT_SUFFIX
    feedback_prompt_template: str = DEFAULT_FEEDBACK_PROMPT_TEMPLATE
    proxy_teacher_template: str = DEFAULT_PROXY_TEACHER_TEMPLATE
    seed: int = 0

    async def __call__(self) -> tuple[FeedbackSelfDistillationDataset, None]:
        tokenizer = get_tokenizer(self.model_name_for_tokenizer)
        renderer = renderers.get_renderer(self.renderer_name, tokenizer=tokenizer)

        train_dataset = FeedbackSelfDistillationDataset(
            batch_size=self.groups_per_batch,
            group_size=self.group_size,
            renderer=renderer,
            tokenizer=tokenizer,
            convo_prefix=self.convo_prefix,
            student_prompt_suffix=self.student_prompt_suffix,
            feedback_prompt_template=self.feedback_prompt_template,
            proxy_teacher_template=self.proxy_teacher_template,
            seed=self.seed,
        )

        # No test dataset for now
        return train_dataset, None
