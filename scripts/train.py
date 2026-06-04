import argparse

def main():
    parser = argparse.ArgumentParser(description="Train multi-task Thai NLP Toolkit model")
    parser.add_argument("--config", type=str, default="./configs/base_config.yaml", help="Path to config file")
    parser.add_argument("--device", type=str, default="cuda", help="Computation device (cpu, cuda, mps)")
    
    args = parser.parse_args()
    print(f"Starting training process with config: {args.config} on device: {args.device}")
    # TODO: Load config, initialize tokenizer, model, datasets, dataloaders, and trainer. Run trainer.train_epoch()

if __name__ == "__main__":
    main()
