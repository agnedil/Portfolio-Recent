"""Train a Gemma-4 model to solve Sudoku puzzles via GRPO reinforcement learning.

Load a Gemma-4 model in 16-bit LoRA mode, define a Sudoku game environment with
sandboxed strategy execution, and train with GRPO using three reward signals
(valid Python, no external imports, puzzle-solving success).
Run with defaults: python gemma4-31b-reinforcement-learning.py. Override e.g.
--model-name unsloth/gemma-4-E4B-it --max-steps -1 --num-train-epochs 1.

The base Gemma-4 weights stay frozen — only the LoRA adapters (attached via
FastVisionModel.get_peft_model, r=32) are updated. GRPO is the RL algorithm used to compute the gradient signal
from the three reward functions, but it's applied as parameter-efficient fine-tuning on top of the pre-trained
model, not training from scratch.                                                                                 

So the pipeline is: pre-trained Gemma-4 → attach LoRA → GRPO updates the LoRA weights using rewards → save
adapters. Same shape as the SFT scripts, just with reward-driven updates instead of supervised loss.

Note on model size: the actual model size fine-tuned is whatever `--model-name` resolves to.
This script defaults to `unsloth/gemma-4-E2B-it` (not 31B), so RL fits
on a single free Colab T4. Pass `--model-name unsloth/gemma-4-31B-it` to
target the 31B variant (requires substantially more VRAM, and GRPO rollouts
will be much slower).
"""

import argparse
import copy
import logging
import random
from dataclasses import dataclass, field
from typing import Callable, List, Optional

import numpy as np
import torch
from datasets import Dataset
from trl import GRPOConfig, GRPOTrainer
from unsloth import (
    FastVisionModel,
    check_python_modules,
    create_locked_down_function,
    execute_with_time_limit,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


DEFAULT_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]

DEFAULT_PROMPT = """
Create a Sudoku solving strategy using only native Python built-in functions without any import statements.
You are given two lists of lists (9x9 grids):
- board: current state (0 means empty)
- initial: starting puzzle (0 means was empty, numbers are fixed)

Return a tuple (row, col, number) for the next move.
- row: 0-8 (row index)
- col: 0-8 (column index)
- number: 1-9 (digit to place)

Only place numbers in cells that are BOTH empty in initial AND empty in board (initial[row][col] == 0 AND board[row][col] == 0)
Use Sudoku rules: no duplicates in rows, columns, or 3x3 boxes.
Output your function in backticks:
```python
def strategy(board, initial):
    # Your logic here
    return (row, col, number)
```
All helper functions must be inside def strategy. Output only the function.
""".strip()


# ---------- Sudoku environment ----------

def _is_valid_placement(board: List[List[int]], row: int, col: int, num: int) -> bool:
    """Check if placing `num` at (row, col) violates Sudoku rules."""
    if num in board[row]:
        return False
    if num in [board[r][col] for r in range(9)]:
        return False
    box_row, box_col = 3 * (row // 3), 3 * (col // 3)
    for r in range(box_row, box_row + 3):
        for c in range(box_col, box_col + 3):
            if board[r][c] == num:
                return False
    return True


def _solve_sudoku(board: List[List[int]]) -> bool:
    """Backtracking solver used to generate complete boards."""
    for row in range(9):
        for col in range(9):
            if board[row][col] == 0:
                for num in range(1, 10):
                    if _is_valid_placement(board, row, col, num):
                        board[row][col] = num
                        if _solve_sudoku(board):
                            return True
                        board[row][col] = 0
                return False
    return True


def _generate_complete_board(rng: random.Random) -> List[List[int]]:
    """Generate a fully-filled valid Sudoku board."""
    board = [[0 for _ in range(9)] for _ in range(9)]
    # Fill diagonal 3x3 boxes first since they don't constrain each other.
    for box in range(3):
        nums = list(range(1, 10))
        rng.shuffle(nums)
        for i in range(3):
            for j in range(3):
                board[box * 3 + i][box * 3 + j] = nums[i * 3 + j]
    _solve_sudoku(board)
    return board


@dataclass
class SudokuGame:
    """Single Sudoku puzzle with move validation and terminal-state tracking."""

    difficulty: int = 40  # Cells to remove (20 easy, 40 medium, 50 hard).
    seed: Optional[int] = None
    _rng: random.Random = field(init=False, repr=False)
    _board: List[List[int]] = field(init=False, repr=False)
    _solution: List[List[int]] = field(init=False, repr=False)
    _initial_board: List[List[int]] = field(init=False, repr=False)
    _moves: int = field(default=0, init=False, repr=False)
    _state: str = field(default="ongoing", init=False, repr=False)

    def __post_init__(self):
        self._rng = random.Random(self.seed)
        complete_board = _generate_complete_board(self._rng)
        self._solution = copy.deepcopy(complete_board)
        self._board = copy.deepcopy(complete_board)
        cells = [(r, c) for r in range(9) for c in range(9)]
        self._rng.shuffle(cells)
        for r, c in cells[: self.difficulty]:
            self._board[r][c] = 0
        self._initial_board = copy.deepcopy(self._board)
        self._update_state()

    def board(self) -> List[List[int]]:
        return [row[:] for row in self._board]

    def initial_board(self) -> List[List[int]]:
        return [row[:] for row in self._initial_board]

    def state(self) -> str:
        return self._state

    def moves(self) -> int:
        return self._moves

    def place_number(self, row: int, col: int, num: int) -> bool:
        """Apply a move; any rule violation puts the game into 'failed' state."""
        if not (0 <= row < 9 and 0 <= col < 9):
            self._state = "failed"
            return False
        if not (1 <= num <= 9):
            self._state = "failed"
            return False
        if self._initial_board[row][col] != 0:
            self._state = "failed"
            return False
        if self._board[row][col] != 0:
            self._state = "failed"
            return False
        if not _is_valid_placement(self._board, row, col, num):
            self._state = "failed"
            return False
        self._board[row][col] = num
        self._moves += 1
        self._update_state()
        return True

    def _update_state(self) -> None:
        if all(self._board[r][c] != 0 for r in range(9) for c in range(9)):
            self._state = "success" if self._board == self._solution else "failed"
        else:
            self._state = "ongoing"


def _execute_strategy(strategy: Callable, game: SudokuGame, max_moves: int = 100):
    """Run a strategy callable repeatedly until success/failure or move limit."""
    assert callable(strategy)
    valid_moves = 0
    while game.state() == "ongoing" and valid_moves < max_moves:
        try:
            result = strategy(game.board(), game.initial_board())
            if not isinstance(result, (tuple, list)) or len(result) != 3:
                return valid_moves, "failed"
            row, col, num = result
            if not all(isinstance(x, int) for x in [row, col, num]):
                return valid_moves, "failed"
            if not game.place_number(row, col, num):
                return valid_moves, "failed"
            valid_moves += 1
        except Exception:
            return valid_moves, "failed"
    if valid_moves >= max_moves and game.state() == "ongoing":
        return valid_moves, "failed"
    return valid_moves, game.state()


@execute_with_time_limit(10)
def execute_strategy(strategy: Callable, game: SudokuGame):
    """Wrap `_execute_strategy` with a 10s wall-clock timeout."""
    return _execute_strategy(strategy, game)


# ---------- Reward functions ----------

def extract_function(text: str) -> Optional[str]:
    """Pull a `def strategy(...)` block out of a fenced Python code block."""
    if text.count("```") >= 2:
        first = text.find("```") + 3
        second = text.find("```", first)
        fx = text[first:second].strip()
        fx = fx.removeprefix("python\n")
        fx = fx[fx.find("def"):]
        if fx.startswith("def strategy(board, initial):"):
            return fx
    return None


def function_works(completions, **kwargs):
    """Reward generations whose code parses and locks down without errors."""
    scores = []
    for completion in completions:
        response = completion[0]["content"]
        function = extract_function(response)
        info = {}
        if function is not None:
            _, info = check_python_modules(function)
        if function is None or "error" in info:
            scores.append(-2.0)
            continue
        try:
            create_locked_down_function(function)
            scores.append(1.0)
        except Exception:
            scores.append(-1.0)
    return scores


def no_cheating(completions, **kwargs):
    """Penalize generations that import third-party modules."""
    scores = []
    for completion in completions:
        response = completion[0]["content"]
        function = extract_function(response)
        if function is None:
            scores.append(-1.0)
            continue
        ok, _ = check_python_modules(function)
        scores.append(1.0 if ok else -20.0)
    return scores


def make_strategy_succeeds(difficulty: int, print_every: int):
    """Build the puzzle-solving reward closure (captures difficulty + print cadence)."""
    counter = {"n": 0}

    def strategy_succeeds(completions, **kwargs):
        scores = []
        seed = int(np.random.randint(10000))
        for completion in completions:
            printed = False
            response = completion[0]["content"]
            function = extract_function(response)

            if counter["n"] % print_every == 0:
                printed = True
                print("\n" + "=" * 60)
                print(function)
                print("=" * 60)
            counter["n"] += 1

            info = {}
            if function is not None:
                _, info = check_python_modules(function)
            if function is None or "error" in info:
                scores.append(0)
                continue

            try:
                new_strategy = create_locked_down_function(function)
            except Exception:
                scores.append(0)
                continue

            try:
                game = SudokuGame(difficulty=difficulty, seed=seed)
                valid_moves, game_state = execute_strategy(new_strategy, game)
                if valid_moves == difficulty:
                    game_state = "success"

                print(f"\n Valid moves: {valid_moves}, Final state: {game_state}")
                if not printed:
                    print("Strategy:")
                    print(function[:200] + "..." if len(function) > 200 else function)

                if game_state == "success":
                    scores.append(30.0)
                elif valid_moves > 0:
                    scores.append(valid_moves * 0.2)
                else:
                    scores.append(-2.0)
            except TimeoutError:
                print("Timeout")
                scores.append(-1.0)
            except Exception as e:
                print(f"Exception: {str(e)[:100]}")
                scores.append(-3.0)
        return scores
    return strategy_succeeds


# ---------- CLI / pipeline ----------

def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for paths, model loading, LoRA, GRPO, and reward setup."""
    parser = argparse.ArgumentParser(
        description="GRPO fine-tune a Gemma-4 model on a Sudoku-solving task.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Model / output paths
    parser.add_argument("--model-name", type=str, default="unsloth/gemma-4-E2B-it")
    parser.add_argument("--output-dir", type=str, default="gemma_4_lora")
    parser.add_argument("--checkpoint-dir", type=str, default="outputs")
    parser.add_argument("--hf-token", type=str, default=None)

    # Model loading
    parser.add_argument("--max-seq-length", type=int, default=4096)
    parser.add_argument(
        "--load-in-4bit",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Default False -> 16-bit LoRA, matching the notebook.",
    )
    parser.add_argument(
        "--fast-inference",
        action=argparse.BooleanOptionalAction,
        default=False,
    )

    # LoRA
    parser.add_argument("--lora-rank", type=int, default=32)
    parser.add_argument(
        "--lora-alpha",
        type=int,
        default=None,
        help="Defaults to 2 * lora_rank when omitted.",
    )
    parser.add_argument(
        "--target-modules",
        type=str,
        nargs="+",
        default=DEFAULT_TARGET_MODULES,
    )
    parser.add_argument("--use-gradient-checkpointing", type=str, default="unsloth")

    # Dataset / prompt
    parser.add_argument("--num-prompts", type=int, default=1000)
    parser.add_argument("--prompt", type=str, default=DEFAULT_PROMPT)

    # Reward / environment
    parser.add_argument("--sudoku-difficulty", type=int, default=40)
    parser.add_argument("--print-every", type=int, default=5)

    # GRPO training hyperparameters
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=0.001)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--lr-scheduler-type", type=str, default="linear")
    parser.add_argument("--optim", type=str, default="adamw_8bit")
    parser.add_argument("--logging-steps", type=int, default=1)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=2)
    parser.add_argument("--num-generations", type=int, default=2)
    parser.add_argument(
        "--max-steps",
        type=int,
        default=60,
        help="Set to -1 to disable and use --num-train-epochs instead.",
    )
    parser.add_argument("--num-train-epochs", type=float, default=1.0)
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument("--epsilon", type=float, default=0.2)
    parser.add_argument("--epsilon-high", type=float, default=0.28)
    parser.add_argument("--delta", type=float, default=1.5)
    parser.add_argument("--loss-type", type=str, default="bnpo")
    parser.add_argument(
        "--mask-truncated-completions",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--report-to", type=str, default="none")

    return parser.parse_args()


def load_model_and_tokenizer(args: argparse.Namespace):
    """Load the base model in 16-bit LoRA mode and attach LoRA adapters."""
    logger.info("Loading base model: %s", args.model_name)
    model, tokenizer = FastVisionModel.from_pretrained(
        model_name=args.model_name,
        max_seq_length=args.max_seq_length,
        load_in_4bit=args.load_in_4bit,
        fast_inference=args.fast_inference,
        token=args.hf_token,
    )

    lora_alpha = args.lora_alpha if args.lora_alpha is not None else args.lora_rank * 2
    logger.info("Attaching LoRA adapters (r=%d, alpha=%d).", args.lora_rank, lora_alpha)
    model = FastVisionModel.get_peft_model(
        model,
        r=args.lora_rank,
        target_modules=args.target_modules,
        lora_alpha=lora_alpha,
        use_gradient_checkpointing=args.use_gradient_checkpointing,
        random_state=args.seed,
    )
    return model, tokenizer


def prepare_dataset(args: argparse.Namespace, tokenizer):
    """Build a 1-prompt dataset replicated `num_prompts` times and return its max prompt length."""
    prompt_text = args.prompt.strip()
    dataset = Dataset.from_list(
        [{"prompt": [{"role": "user", "content": prompt_text}], "answer": 0}]
        * args.num_prompts
    )
    maximum_length = len(
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt_text}],
            add_generation_prompt=True,
        )
    )
    logger.info("Maximum prompt length: %d", maximum_length)
    return dataset, maximum_length


def build_trainer(
    args: argparse.Namespace, model, tokenizer, dataset, maximum_length: int
) -> GRPOTrainer:
    """Build a GRPOTrainer with the three Sudoku reward functions."""
    # Reserve room for the prompt (plus 1-token safety margin) within max_seq_length.
    max_completion_length = args.max_seq_length - (maximum_length + 1)

    training_args = GRPOConfig(
        temperature=args.temperature,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type=args.lr_scheduler_type,
        optim=args.optim,
        logging_steps=args.logging_steps,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_generations=args.num_generations,
        max_completion_length=max_completion_length,
        max_steps=args.max_steps,
        num_train_epochs=args.num_train_epochs,
        save_steps=args.save_steps,
        report_to=args.report_to,
        output_dir=args.checkpoint_dir,
        epsilon=args.epsilon,
        epsilon_high=args.epsilon_high,
        delta=args.delta,
        loss_type=args.loss_type,
        mask_truncated_completions=args.mask_truncated_completions,
        seed=args.seed,
    )

    return GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[
            function_works,
            no_cheating,
            make_strategy_succeeds(args.sudoku_difficulty, args.print_every),
        ],
        args=training_args,
        train_dataset=dataset,
    )


def log_memory_stats(prefix: str, baseline_gb: float | None = None) -> float:
    """Log peak reserved GPU memory; return it so callers can diff before/after training."""
    if not torch.cuda.is_available():
        return 0.0
    used_gb = round(torch.cuda.max_memory_reserved() / 1024**3, 3)
    total_gb = round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 3)
    logger.info("%s: peak reserved %.3f / %.3f GB", prefix, used_gb, total_gb)
    if baseline_gb is not None:
        logger.info("%s: peak attributable to training %.3f GB", prefix, used_gb - baseline_gb)
    return used_gb


def main() -> None:
    """End-to-end pipeline: load -> build dataset -> GRPO train -> save adapters."""
    args = parse_args()

    model, tokenizer = load_model_and_tokenizer(args)
    dataset, maximum_length = prepare_dataset(args, tokenizer)
    trainer = build_trainer(args, model, tokenizer, dataset, maximum_length)

    start_mem = log_memory_stats("Pre-training memory")

    logger.info("Starting GRPO training.")
    stats = trainer.train()
    logger.info(
        "Training finished in %.2f minutes.",
        stats.metrics["train_runtime"] / 60,
    )

    log_memory_stats("Post-training memory", baseline_gb=start_mem)

    logger.info("Saving LoRA adapters and tokenizer to: %s", args.output_dir)
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    logger.info("Done.")


if __name__ == "__main__":
    main()
