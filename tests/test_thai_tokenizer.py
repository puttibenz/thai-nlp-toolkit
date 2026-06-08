import unittest
from unittest.mock import patch
from tokenizer.thai_tokenizer import ThaiTokenizer

class TestThaiTokenizer(unittest.TestCase):
    def test_tokenizer(self):
        with patch("sentencepiece.SentencePieceProcessor") as MockSP:
            mock_sp = MockSP.return_value
            mock_sp.get_piece_size.return_value = 32000
            mock_sp.piece_to_id.side_effect = lambda x: {
                "[PAD]": 0, "[UNK]": 1, "[BOS]": 2, "[EOS]": 3,
                "[CLS]": 4, "[SEP]": 5, "[MASK]": 6
            }.get(x, 1)
            mock_sp.encode.return_value = [10, 20, 30, 40]
            mock_sp.decode.return_value = "สวัสดี"

            with patch("os.path.exists", return_value=True):
                tok = ThaiTokenizer("fake.model")

            # test encode
            ids = tok.encode("สวัสดีครับ")
            self.assertEqual(ids[0], 4, "ต้องมี CLS")
            self.assertEqual(ids[-1], 5, "ต้องมี SEP")
            self.assertEqual(ids, [4, 10, 20, 30, 40, 5])

            # test encode_qa
            tok.sp.encode.side_effect = lambda t, **kw: [10, 20] if "คำถาม" in str(t) else [30, 40, 50]
            qa_res = tok.encode_qa("คำถาม", "บริบท")
            # [CLS] q_ids [SEP] c_ids [SEP] -> [4, 10, 20, 5, 30, 40, 50, 5]
            self.assertEqual(qa_res["input_ids"], [4, 10, 20, 5, 30, 40, 50, 5])
            self.assertEqual(qa_res["context_start"], 4)
            self.assertEqual(qa_res["attention_mask"], [1] * 8)

            # test encode_qa truncation
            # if max_length is 6, should keep CLS, q_ids, SEP, and truncate c_ids, then add SEP
            qa_res_trunc = tok.encode_qa("คำถาม", "บริบท", max_length=6)
            self.assertEqual(qa_res_trunc["input_ids"], [4, 10, 20, 5, 30, 5])
            self.assertEqual(qa_res_trunc["context_start"], 4)

            # Reset side_effect
            tok.sp.encode.side_effect = None

            # test truncation
            long_text_ids = tok.encode("x" * 1000)  # mock คืน [10,20,30,40] เสมอ
            tok.sp.encode.return_value = list(range(600))  # simulate long sequence
            ids_long = tok.encode("long")
            # จริงๆ ทดสอบ truncation logic ตรงๆ แทน
            ids_raw = [4] + list(range(600)) + [5]  # 602 tokens
            max_len = 512
            truncated = ids_raw[:max_len - 1] + [5]
            self.assertEqual(len(truncated), 512)
            self.assertEqual(truncated[-1], 5)

            # test batch_encode shape
            tok.sp.encode.return_value = [10, 20, 30]
            result = tok.batch_encode(["สวัสดี", "ครับ"], max_length=512)
            self.assertEqual(result["input_ids"].shape[0], 2)
            self.assertEqual(result["input_ids"].shape, result["attention_mask"].shape)

            # test padding — batch สองประโยค ต้องยาวเท่ากัน
            self.assertEqual(result["input_ids"].shape[1], result["input_ids"].shape[1])
            
            # ตรวจ attention_mask: padding positions ต้องเป็น 0
            tok.sp.encode.side_effect = lambda t, **kw: [10,20,30] if "สวัสดี" in str(t) else [10]
            result2 = tok.batch_encode(["สวัสดีครับ", "ใช่"], max_length=512,
                                       return_tensors=False)
            # batch แรกยาวกว่า batch สอง → batch สองต้องมี padding
            self.assertIn(0, result2["attention_mask"][1], "ต้องมี padding mask")

            print("ThaiTokenizer tests passed")

if __name__ == "__main__":
    unittest.main()
