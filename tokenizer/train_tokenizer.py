import os
import sys
import argparse
import logging
import sentencepiece as spm
import pathlib 

# Configure UTF-8 encoding for standard output/error on Windows
if sys.platform.startswith("win"):
    if sys.stdout.encoding != 'utf-8':
        try:
            sys.stdout.reconfigure(encoding='utf-8')
        except AttributeError:
            pass
    if sys.stderr.encoding != 'utf-8':
        try:
            sys.stderr.reconfigure(encoding='utf-8')
        except AttributeError:
            pass

root = pathlib.Path(__file__).resolve().parent
while root.parent != root:
    if (root / "requirements.txt").exists() or (root / "README.md").exists():
        sys.path.append(str(root))
        break
    root = root.parent

from tokenizer.preprocess import preprocess_thai, preprocess_file

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train SentencePiece BPE tokenizer สำหรับ Thai NLP Toolkit"
    )
    parser.add_argument(
        "--corpus", type=str, required=True,
        help="Path ของ raw corpus file (1 บรรทัด = 1 document)"
    )
    parser.add_argument(
        "--output_dir", type=str, default="./tokenizer",
        help="Directory สำหรับเก็บ output ทั้งหมด"
    )
    parser.add_argument(
        "--vocab_size", type=int, default=32000,
        help="จำนวน vocab (default 32000)"
    )
    parser.add_argument(
        "--character_coverage", type=float, default=0.9995,
        help="Coverage ของ characters ใน vocab (default 0.9995 สำหรับภาษาไทย)"
    )
    parser.add_argument(
        "--model_prefix", type=str, default="thai_bpe",
        help="ชื่อ prefix ของ output model file"
    )
    parser.add_argument(
        "--min_length", type=int, default=10,
        help="ความยาวขั้นต่ำของ line หลัง preprocess (default 10 chars)"
    )
    parser.add_argument(
        "--skip_preprocess", action="store_true",
        help="ข้าม preprocess step ถ้า corpus clean แล้ว"
    )
    return parser.parse_args()


def validate_corpus(path: str) -> int:
    """ตรวจสอบว่า corpus ใช้ได้ และ return จำนวน lines"""
    if not os.path.exists(path):
        raise FileNotFoundError(f"ไม่พบ corpus file: {path}")

    size_mb = os.path.getsize(path) / (1024 ** 2)
    log.info(f"corpus size: {size_mb:.1f} MB")

    if size_mb < 1:
        log.warning("corpus เล็กมาก (<1MB) — vocab อาจไม่ครอบคลุมพอ แนะนำอย่างน้อย 100MB")

    with open(path, encoding="utf-8", errors="replace") as f:
        n_lines = sum(1 for _ in f)

    log.info(f"corpus lines: {n_lines:,}")

    if n_lines < 10_000:
        log.warning("corpus มี lines น้อยมาก — ควรมีอย่างน้อย 1M lines สำหรับ vocab_size=32000")

    return n_lines


def preprocess_corpus(input_path: str, output_path: str, min_length: int) -> int:
    """
    Preprocess corpus ทีละบรรทัด
    return จำนวน lines ที่เก็บไว้หลัง filter
    """
    log.info(f"preprocessing: {input_path} → {output_path}")
    kept = preprocess_file(input_path, output_path, min_length=min_length)
    log.info(f"kept {kept:,} lines หลัง filter")
    return kept


def train_sentencepiece(
    corpus_path: str,
    output_dir: str,
    model_prefix: str,
    vocab_size: int,
    character_coverage: float,
) -> str:
    """
    Train SentencePiece BPE model
    return path ของ .model file ที่ได้
    """
    model_path = os.path.join(output_dir, model_prefix)

    log.info(f"training SentencePiece BPE...")
    log.info(f"  vocab_size={vocab_size}, character_coverage={character_coverage}")

    spm.SentencePieceTrainer.train(
        # ── Input ──────────────────────────────────
        input=corpus_path,
        input_sentence_size=1_000_000,   # จำกัด lines ที่ใช้ train (ป้องกัน OOM และลดเวลาการเทรนลงมาก)
        shuffle_input_sentence=True,     # shuffle ก่อน train เพื่อ distribution ดีขึ้น

        # ── Output ─────────────────────────────────
        model_prefix=model_path,
        vocab_size=vocab_size,
        model_type="bpe",

        # ── Coverage ───────────────────────────────
        character_coverage=character_coverage,
        # 0.9995 เหมาะกับภาษาที่มี unicode เยอะ เช่น ไทย
        # ถ้าใช้ค่าสูงกว่านี้ vocab จะโตโดยไม่จำเป็น

        # ── Special tokens ─────────────────────────
        # ลำดับสำคัญ: pad=0, unk=1, bos=2, eos=3
        pad_id=0,  unk_id=1,  bos_id=2,  eos_id=3,
        pad_piece="[PAD]", unk_piece="[UNK]",
        bos_piece="[BOS]", eos_piece="[EOS]",
        # user_defined_symbols จะได้ id 4, 5, 6 ตามลำดับ
        user_defined_symbols=["[CLS]", "[SEP]", "[MASK]"],

        # ── Normalization ───────────────────────────
        normalization_rule_name="nfkc",
        # nfkc normalize fullwidth → ASCII ใน SentencePiece layer ด้วย
        # เป็น safety net แม้ preprocess.py จัดการแล้ว

        # ── Subword regularization ──────────────────
        # ช่วยให้ model robust ต่อ segmentation ที่ต่างกัน
        byte_fallback=True,
        # ถ้าเจอ char ที่ไม่อยู่ใน vocab จะ fallback เป็น UTF-8 bytes
        # แทนที่จะใช้ [UNK] ทำให้ round-trip decode ถูกต้องเสมอ

        # ── Performance ────────────────────────────
        num_threads=os.cpu_count(),
        train_extremely_large_corpus=True,   # ใช้ disk แทน RAM ถ้า corpus ใหญ่
    )

    model_file = model_path + ".model"
    vocab_file = model_path + ".vocab"
    log.info(f"saved model → {model_file}")
    log.info(f"saved vocab → {vocab_file}")
    return model_file


def verify_tokenizer(model_file: str) -> None:
    """
    Quick sanity check หลัง train เสร็จ
    ตรวจว่า encode → decode round-trip ถูกต้อง
    """
    from tokenizer.thai_tokenizer import ThaiTokenizer

    log.info("verifying tokenizer...")
    tok = ThaiTokenizer(model_file)

    test_cases = [
        "สวัสดีครับ ผมชื่อสมชาย",
        "Bangkok is the capital of Thailand",
        "ราคา 1,234.56 บาท",
        "ทดสอบ test 123 ทดสอบ",
    ]

    all_ok = True
    for text in test_cases:
        ids   = tok.encode(text, add_special_tokens=False)
        recon = tok.decode(ids)
        ok    = recon.strip() == preprocess_thai(text).strip()
        status = "✓" if ok else "✗"
        log.info(f"  {status} '{text[:30]}' → {len(ids)} tokens → '{recon[:30]}'")
        if not ok:
            all_ok = False

    # ตรวจ special tokens
    assert tok.pad_id  == 0, f"PAD id ผิด: {tok.pad_id}"
    assert tok.bos_id  == 2, f"BOS id ผิด: {tok.bos_id}"
    assert tok.eos_id  == 3, f"EOS id ผิด: {tok.eos_id}"
    assert tok.vocab_size == tok.sp.get_piece_size()
    log.info(f"  vocab_size = {tok.vocab_size:,}")

    if all_ok:
        log.info("tokenizer OK ✓")
    else:
        log.warning("round-trip มีบางกรณีที่ไม่ตรง — ตรวจสอบ preprocess ด้วย")


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # ── Step 1: Validate ────────────────────────────
    n_lines = validate_corpus(args.corpus)

    # ── Step 2: Preprocess ──────────────────────────
    if args.skip_preprocess:
        log.info("ข้าม preprocess (--skip_preprocess)")
        clean_corpus = args.corpus
    else:
        clean_corpus = os.path.join(args.output_dir, "corpus_clean.txt")
        kept = preprocess_corpus(args.corpus, clean_corpus, args.min_length)

        if kept < 1000:
            log.error(f"หลัง filter เหลือแค่ {kept} lines — ตรวจสอบ corpus หรือลด --min_length")
            sys.exit(1)

    # ── Step 3: Train SentencePiece ─────────────────
    model_file = train_sentencepiece(
        corpus_path=clean_corpus,
        output_dir=args.output_dir,
        model_prefix=args.model_prefix,
        vocab_size=args.vocab_size,
        character_coverage=args.character_coverage,
    )

    # ── Step 4: Verify ──────────────────────────────
    verify_tokenizer(model_file)

    log.info("=" * 50)
    log.info("training complete!")
    log.info(f"model: {model_file}")
    log.info(f"usage: ThaiTokenizer('{model_file}')")


if __name__ == "__main__":
    main()