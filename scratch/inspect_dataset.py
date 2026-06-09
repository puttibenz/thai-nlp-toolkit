from datasets import load_dataset
try:
    ds = load_dataset("pythainlp/thainer-corpus-v2.2", trust_remote_code=True)
    print("Features:", ds["train"].features)
    print("Example:", ds["train"][0])
except Exception as e:
    print("Error:", e)
