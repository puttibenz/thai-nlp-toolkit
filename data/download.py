import argparse
import os

def download_ner():
    print("Downloading BEST2020 NER dataset...")
    # TODO: Fetch and unpack BEST2020

def download_sentiment():
    print("Downloading Wisesight Sentiment dataset...")
    # TODO: Fetch and unpack Wisesight Sentiment

def download_qa():
    print("Downloading iApp Thai QA dataset...")
    # TODO: Fetch and unpack iApp Thai QA

def download_wikipedia():
    print("Downloading ThaiWiki dump for pre-training vocab...")
    # TODO: Fetch and unpack Thai Wikipedia dump

def main():
    parser = argparse.ArgumentParser(description="Download datasets for Thai NLP Toolkit")
    parser.add_argument("--task", type=str, default="all", choices=["ner", "sentiment", "qa", "vocab", "all"],
                        help="Task-specific dataset to download")
    parser.add_argument("--output_dir", type=str, default="./data/raw", help="Output directory")

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    if args.task in ["ner", "all"]:
        download_ner()
    if args.task in ["sentiment", "all"]:
        download_sentiment()
    if args.task in ["qa", "all"]:
        download_qa()
    if args.task in ["vocab", "all"]:
        download_wikipedia()

if __name__ == "__main__":
    main()
