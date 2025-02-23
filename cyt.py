import argparse
import logging
import os

from utils import Evaluator, load_datasets
from trl import GRPOConfig, GRPOTrainer
from transformers import AutoModelForCausalLM
from bitsandbytes.optim import Adam8bit
from datasets import Dataset

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('cott.log'),
            logging.StreamHandler()
        ]
    )

def train(args):
    evaluator = Evaluator(
        max_clength=args.maxl,
        min_clength=args.minl,
        stepsize=args.step_size)
    ds = load_datasets(args.datasets)
    ds = ds.add_column("prompt", [[{'role': 'system', 'content': "Please reason step by step, and put your final answer within \\boxed{{}}."},
                                  {'role': 'user', 'content': e['problem']}] for e in ds])
    training_args = GRPOConfig(
        output_dir="./logs",
        logging_steps=10,
        max_completion_length=args.maxl,
        per_device_train_batch_size=args.per_device_batch_size,
        num_generations=args.num_generations,
        save_steps=args.save_steps,
        num_train_epochs=args.epoch,
        bf16=True,
        # use_vllm=True
    )

    model = AutoModelForCausalLM.from_pretrained(args.model)
    optimizer = Adam8bit(model.parameters(), lr=training_args.learning_rate)

    trainer = GRPOTrainer(
        model = model,
        reward_funcs=evaluator,
        args=training_args,
        train_dataset=ds
    )
    trainer.optimizer = optimizer
    trainer.train()

def main():
    setup_logging()
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    parser = argparse.ArgumentParser(description="compress your thought with rl")
    parser.add_argument('-m', '--model', type=str, default='hkust-nlp/Qwen-2.5-Math-7B-SimpleRL-Zero', help="This argument specifies the model path")
    parser.add_argument('--step_size', type=int, default=2, help="the decrease step size of the completion length")
    parser.add_argument('--save_steps', type=int, default=160, help="The save steps in the training process")
    parser.add_argument('--maxl', type=int, default=1024, help="The maximum step_size of the completion length")
    parser.add_argument('--minl', type=int, default=512, help="The minimum step_size of the completion length")
    parser.add_argument('-d', '--datasets', nargs='+', default=["datasets/math500.jsonl"], help="The training dataset used in this program")
    parser.add_argument('-b', '--per_device_batch_size', type=int, default=2, help="The train batch size per device")
    parser.add_argument('-n', '--num_generations', type=int, default=4, help="The number of generations per input prompt")
    parser.add_argument('-e', '--epoch', type=float, default=2.0, help="The training epochs of the datasets")

    args = parser.parse_args()

    train(args)

if __name__ == "__main__":
    main()
