import sys
import argparse

def main():
    parser = argparse.ArgumentParser(description="Train custom BPE tokenizer for Thai NLP Toolkit")
    parser.add_argument("--corpus", type=str, required=True, help="Path to corpus file")
    parser.add_argument("--vocab_size", type=int, default=32000, help="Vocabulary size")
    parser.add_argument("--output", type=str, default="./tokenizer/thai_bpe", help="Prefix for output model/vocab files")
    
    args = parser.parse_args()
    print(f"Training tokenizer with corpus: {args.corpus}, vocab_size: {args.vocab_size}")
    # TODO: Load corpus, pre-tokenize, train sentencepiece, and save

if __name__ == "__main__":
    main()
