import argparse
import logging
import os

from utils import Evaluator, load_datasets
from trl import GRPOConfig, GRPOTrainer
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
        max_completion_length=1024,
        per_device_train_batch_size=2,
        bf16=True,
        num_generations=4,
        save_steps=160,
        num_train_epochs=3.0,
    )
    trainer = GRPOTrainer(
        model = args.model,
        reward_funcs=evaluator,
        args=training_args,
        train_dataset=ds
    )
    trainer.train()

def main():
    setup_logging()
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    parser = argparse.ArgumentParser(description="compress your thought with rl")
    parser.add_argument('-m', '--model', type=str, default='hkust-nlp/Qwen-2.5-Math-7B-SimpleRL-Zero', help="This argument specifies the model path")
    parser.add_argument('--step_size', type=int, default=2, help="the decrease step size of the completion length")
    parser.add_argument('--maxl', type=int, default=1024, help="The maximum step_size of the completion length")
    parser.add_argument('--minl', type=int, default=512, help="The minimum step_size of the completion length")
    parser.add_argument('-d', '--datasets', nargs='+', default=["datasets/math500.jsonl"], help="The training dataset used in this program")

    args = parser.parse_args()

    train(args)

if __name__ == "__main__":
    main()
