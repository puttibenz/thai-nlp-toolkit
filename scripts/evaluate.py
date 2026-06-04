import argparse

def main():
    parser = argparse.ArgumentParser(description="Evaluate multi-task Thai NLP Toolkit model")
    parser.add_argument("--model_dir", type=str, required=True, help="Directory containing trained model and config")
    parser.add_argument("--task", type=str, default="all", choices=["ner", "sentiment", "qa", "all"],
                        help="Specific task to evaluate")
    parser.add_argument("--device", type=str, default="cuda", help="Computation device (cpu, cuda, mps)")

    args = parser.parse_args()
    print(f"Evaluating task: {args.task} for model: {args.model_dir} on device: {args.device}")
    # TODO: Load model, setup evaluate datasets and run evaluations

if __name__ == "__main__":
    main()
