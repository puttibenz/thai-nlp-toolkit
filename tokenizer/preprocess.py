import re
import unicodedata


# Thai block: U+0E00–U+0E7F
# เก็บ: Thai, Latin (upper+lower), digits, whitespace, วรรคตอนพื้นฐาน
_KEEP_PATTERN = re.compile(
    r"[^\u0E00-\u0E7F"   # Thai
    r"a-zA-Z"            # Latin
    r"0-9"               # digits
    r"\s"                # whitespace
    r"\.\,\!\?\(\)\-\:\;\"\'\/"  # วรรคตอน
    r"]"
)

# Zero-width characters ที่พบบ่อยใน Thai web text
_ZERO_WIDTH = re.compile(
    r"[\u200B"   # Zero-width space (ZWSP) — พบใน Wikipedia Thai มาก
    r"\u200C"    # Zero-width non-joiner (ZWNJ)
    r"\u200D"    # Zero-width joiner (ZWJ)
    r"\uFEFF"    # BOM
    r"\u00AD"    # Soft hyphen
    r"]"
)

# Fullwidth Latin และ digits → ASCII
_FULLWIDTH_MAP = str.maketrans(
    "！＂＃＄％＆＇（）＊＋，－．／"
    "０１２３４５６７８９"
    "：；＜＝＞？＠"
    "ＡＢＣＤＥＦＧＨＩＪＫＬＭＮＯＰＱＲＳＴＵＶＷＸＹＺ"
    "［＼］＾＿｀"
    "ａｂｃｄｅｆｇｈｉｊｋｌｍｎｏｐｑｒｓｔｕｖｗｘｙｚ"
    "｛｜｝～",
    "!\"#$%&'()*+,-./"
    "0123456789"
    ":;<=>?@"
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "[\\]^_`"
    "abcdefghijklmnopqrstuvwxyz"
    "{|}~"
)

# Thai digits → Arabic
_THAI_DIGITS = str.maketrans("๐๑๒๓๔๕๖๗๘๙", "0123456789")


def preprocess_thai(text: str) -> str:
    """
    Clean Thai text ก่อนส่งเข้า SentencePiece training หรือ inference

    ลำดับสำคัญ — อย่าสลับขั้นตอน:
    1. NFC ก่อน เพราะ regex ที่ใช้ codepoint range จะทำงานถูกต้องหลัง normalize เท่านั้น
    2. Zero-width ก่อน noise อื่น เพราะบางครั้ง ZWSP อยู่ติดกับ HTML entity
    3. HTML/URL ก่อน fullwidth เพราะ URL บางตัวมี fullwidth chars
    4. Thai-specific หลัง noise เพราะต้องการ text ที่สะอาดแล้ว
    5. Whitespace สุดท้ายเสมอ
    """
    if not text or not text.strip():
        return ""

    # 1. Unicode normalization
    text = unicodedata.normalize("NFC", text)

    # 2. Zero-width characters
    text = _ZERO_WIDTH.sub("", text)

    # 3. HTML entities และ tags
    text = re.sub(r"<[^>]{1,100}>", " ", text)          # HTML tags (จำกัด length ป้องกัน ReDoS)
    text = re.sub(r"&[a-zA-Z]{2,8};", " ", text)        # named entities: &nbsp; &amp;
    text = re.sub(r"&#\d{1,6};", " ", text)             # numeric entities: &#160;

    # 4. URLs และ emails
    text = re.sub(r"https?://\S{1,500}", " ", text)
    text = re.sub(r"www\.\S{1,500}", " ", text)
    text = re.sub(r"\S{1,100}@\S{1,100}\.\S{2,10}", " ", text)

    # 5. Fullwidth → ASCII
    text = text.translate(_FULLWIDTH_MAP)

    # 6. Thai digits → Arabic
    text = text.translate(_THAI_DIGITS)

    # 7. Thai-specific: "เเ" (2×sara e) → "แ" (sara ae)
    # เป็น bug ที่พบบ่อยมากจากการ type บน keyboard Thai
    text = text.replace("\u0E40\u0E40", "\u0E41")

    # 8. ลบ tone marks ซ้ำ (เช่น ้้ หรือ ่่)
    # Thai tone marks: U+0E48-U+0E4B
    text = re.sub(r"([\u0E48-\u0E4B])\1+", r"\1", text)

    # 9. Character whitelist — ลบ character ที่ไม่ต้องการ
    text = _KEEP_PATTERN.sub(" ", text)

    # 10. Normalize whitespace — ทำเป็น step สุดท้ายเสมอ
    text = re.sub(r"[ \t]+", " ", text)       # multiple spaces → single
    text = re.sub(r"\n{3,}", "\n\n", text)    # max 2 newlines ติดกัน
    text = text.strip()

    return text


def preprocess_file(input_path: str, output_path: str, min_length: int = 10) -> int:
    """
    Process ทั้งไฟล์ corpus ทีละบรรทัด
    return จำนวน lines ที่เก็บไว้
    """
    kept = 0
    with open(input_path, encoding="utf-8") as fin, \
         open(output_path, "w", encoding="utf-8") as fout:
        for line in fin:
            clean = preprocess_thai(line)
            # กรองบรรทัดสั้นเกินไปออก
            if len(clean) >= min_length:
                fout.write(clean + "\n")
                kept += 1
    return kept

if __name__ == "__main__":
    cases = [
        # (input, expected_output, description)
        ("สวัสดี\u200Bครับ",      "สวัสดีครับ",    "ลบ ZWSP"),
        ("<p>ข้อความ</p>",        "ข้อความ",       "ลบ HTML tags"),
        ("ดู https://example.com ด้วย", "ดู ด้วย",  "ลบ URL"),
        ("Ａ Ｂ Ｃ ๑๒๓",          "A B C 123",     "fullwidth + Thai digits"),
        ("เเมว",                  "แมว",           "เเ → แ"),
        ("ไม้้้โท",               "ไม้โท",         "tone mark ซ้ำ"),
        ("&nbsp;&amp;",           "",              "HTML entities"),
        ("",                      "",              "empty string"),
    ]

    for text, expected, desc in cases:
        result = preprocess_thai(text)
        status = "✓" if result == expected else "✗"
        print(f"{status} {desc}: {repr(result)}")
        if result != expected:
            print(f"  expected: {repr(expected)}")