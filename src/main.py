from src.training.run_experiment import run_experiment
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/base.yaml")
    args = parser.parse_args()

    run_experiment(args.config)


if __name__ == "__main__":
    main()
