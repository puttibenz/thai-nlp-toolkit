# data/download.py
from datasets import load_dataset
import json, os, csv

def download_all(output_dir="data/raw"):
    os.makedirs(output_dir, exist_ok=True)
    download_ner(output_dir)
    download_sentiment(output_dir)
    download_qa(output_dir)

# ── NER ──────────────────────────────────────────────────────────────────────

def download_ner(output_dir="data/raw"):
    print("downloading Thai NER 2.2...")
    ds = load_dataset("pythainlp/thainer-corpus-v2.2", trust_remote_code=True)
    ner_feature = ds["train"].features["ner"].feature

    for split, filename in [("train","ner_train.jsonl"),
                             ("validation","ner_val.jsonl"),
                             ("test","ner_test.jsonl")]:
        path = os.path.join(output_dir, filename)
        with open(path, "w", encoding="utf-8") as f:
            for item in ds[split]:
                # แปลงเป็น format ที่ NERDataset ของเราอ่านได้
                out = {
                    "tokens":   item["words"],
                    "ner_tags": [ner_feature.int2str(tag) for tag in item["ner"]],
                }
                f.write(json.dumps(out, ensure_ascii=False) + "\n")
        print(f"  saved {len(ds[split])} examples -> {path}")

# ── Sentiment ─────────────────────────────────────────────────────────────────

def download_sentiment(output_dir="data/raw"):
    print("downloading Wisesight Sentiment...")
    ds = load_dataset("pythainlp/wisesight_sentiment", trust_remote_code=True)
    cat_feature = ds["train"].features["category"]

    # label mapping: wisesight ใช้ pos/neu/neg/q — เราตัด q ออก
    label_map = {"pos": "pos", "neu": "neu", "neg": "neg"}

    for split, filename in [("train","sent_train.tsv"),
                             ("validation","sent_val.tsv"),
                             ("test","sent_test.tsv")]:
        path = os.path.join(output_dir, filename)
        kept = 0
        with open(path, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f, delimiter="\t")
            for item in ds[split]:
                label = cat_feature.int2str(item["category"])
                if label not in label_map:
                    continue   # ตัด label "q" (question) ออก
                text = item["texts"].replace("\t", " ").replace("\n", " ")
                writer.writerow([text, label_map[label]])
                kept += 1
        print(f"  saved {kept} examples -> {path}")

# ── QA ───────────────────────────────────────────────────────────────────────

def download_qa(output_dir="data/raw"):
    print("downloading iApp Thai QA (SQuAD format)...")
    ds = load_dataset("wannaphong/iapp_wiki_qa_squad", trust_remote_code=True)

    for split, filename in [("train","qa_train.json"),
                             ("validation","qa_val.json"),
                             ("test","qa_test.json")]:
        path = os.path.join(output_dir, filename)
        examples = []
        for item in ds[split]:
            examples.append({
                "question": item["question"],
                "context":  item["context"],
                "answers":  item["answers"]["text"],   # list of answer strings
            })
        with open(path, "w", encoding="utf-8") as f:
            json.dump(examples, f, ensure_ascii=False, indent=2)
        print(f"  saved {len(examples)} examples -> {path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="all",
                        choices=["all","ner","sentiment","qa"])
    parser.add_argument("--output_dir", default="data/raw")
    args = parser.parse_args()

    if args.task == "all":       download_all(args.output_dir)
    elif args.task == "ner":     download_ner(args.output_dir)
    elif args.task == "sentiment": download_sentiment(args.output_dir)
    elif args.task == "qa":      download_qa(args.output_dir)