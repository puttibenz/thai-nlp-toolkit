from datasets import load_dataset
try:
    print("Loading wannaphong/iapp_wiki_qa_squad...")
    ds = load_dataset("wannaphong/iapp_wiki_qa_squad")
    print("QA train features:", ds["train"].features)
    print("QA train example:", ds["train"][0])
except Exception as e:
    print("QA Error:", e)
