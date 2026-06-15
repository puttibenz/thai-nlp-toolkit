# data/download_corpus.py
"""
ดาวน์โหลด Thai Wikipedia corpus จาก Hugging Face
สำหรับใช้เทรน SentencePiece Tokenizer ด้วย vocab_size = 32,000

Usage:
    python -m data.download_corpus
    python -m data.download_corpus --max_lines 2000000
"""
import os
import argparse
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def download_thai_wiki_corpus(output_dir: str, max_lines: int = 2_000_000) -> int:
    """
    ดาวน์โหลด Thai Wikipedia จาก Hugging Face แล้วเขียนเป็น plain text
    1 บรรทัด = 1 ย่อหน้า (paragraph)

    Returns จำนวนบรรทัดที่เขียนได้
    """
    from datasets import load_dataset

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "thai_corpus.txt")

    log.info("กำลังดาวน์โหลด Thai Wikipedia corpus จาก Hugging Face...")
    log.info("(อาจใช้เวลาสักครู่ในการดาวน์โหลดครั้งแรก)")

    # ใช้ streaming เพื่อไม่ให้ RAM เต็ม
    ds = load_dataset(
        "pythainlp/thwiki-20240801",
        split="train",
        streaming=True,
        trust_remote_code=True,
    )

    written = 0
    with open(output_path, "w", encoding="utf-8") as f:
        for item in ds:
            text = item.get("text", "")
            if not text or len(text.strip()) < 20:
                continue

            # แยกย่อหน้า — แต่ละย่อหน้าเป็น 1 บรรทัด
            for paragraph in text.split("\n"):
                paragraph = paragraph.strip()
                if len(paragraph) < 20:
                    continue

                f.write(paragraph + "\n")
                written += 1

                if written % 100_000 == 0:
                    log.info(f"  เขียนไปแล้ว {written:,} บรรทัด...")

                if written >= max_lines:
                    log.info(f"ถึงจำนวนสูงสุด {max_lines:,} บรรทัด — หยุดดาวน์โหลด")
                    break

            if written >= max_lines:
                break

    size_mb = os.path.getsize(output_path) / (1024 ** 2)
    log.info(f"✅ เสร็จสิ้น! เขียนทั้งหมด {written:,} บรรทัด ({size_mb:.1f} MB)")
    log.info(f"   ไฟล์: {output_path}")
    return written


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="ดาวน์โหลด Thai Wikipedia corpus สำหรับเทรน Tokenizer"
    )
    parser.add_argument(
        "--output_dir", default="data/corpus",
        help="โฟลเดอร์สำหรับเก็บไฟล์ corpus (default: data/corpus)"
    )
    parser.add_argument(
        "--max_lines", type=int, default=2_000_000,
        help="จำนวนบรรทัดสูงสุดที่จะดาวน์โหลด (default: 2,000,000)"
    )
    args = parser.parse_args()

    download_thai_wiki_corpus(args.output_dir, args.max_lines)
