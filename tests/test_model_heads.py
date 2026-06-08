import unittest
import torch
from model.heads.ner_head import NERHead
from model.heads.sentiment_head import SentimentHead
from model.heads.qa_head import QAHead

class TestModelHeads(unittest.TestCase):
    def setUp(self):
        self.B = 2
        self.T = 16
        self.D = 64
        # Create input hidden states and attention mask
        self.hidden_states = torch.randn(self.B, self.T, self.D)
        self.attention_mask = torch.ones(self.B, self.T, dtype=torch.long)
        self.attention_mask[1, 10:] = 0  # Batch 1 has padding

    def test_ner_head(self):
        num_labels = 7
        ner_head = NERHead(d_model=self.D, num_labels=num_labels)
        ner_head.eval()
        logits = ner_head(self.hidden_states)
        
        # Verify shape (B, T, num_labels)
        self.assertEqual(logits.shape, (self.B, self.T, num_labels))
        self.assertFalse(torch.isnan(logits).any(), "NER logits should not contain NaNs")

    def test_sentiment_head_cls_pooling(self):
        num_classes = 3
        sent_head = SentimentHead(d_model=self.D, num_classes=num_classes)
        sent_head.eval()
        
        # Run forward pass
        logits = sent_head(self.hidden_states, self.attention_mask)
        
        # Verify shape (B, num_classes)
        self.assertEqual(logits.shape, (self.B, num_classes))
        self.assertFalse(torch.isnan(logits).any(), "Sentiment logits should not contain NaNs")
        
        # Verify that only the CLS token (index 0) affects the logits.
        # If we change non-CLS tokens (indices 1 to T-1), the output should remain exactly the same.
        hidden_states_altered = self.hidden_states.clone()
        hidden_states_altered[:, 1:, :] += 10.0  # Add large values to all but index 0
        
        logits_altered = sent_head(hidden_states_altered, self.attention_mask)
        
        # Output should be identical because only index 0 is pooled
        torch.testing.assert_close(logits, logits_altered, msg="SentimentHead should only look at CLS token (index 0)")

    def test_qa_head_masking(self):
        qa_head = QAHead(d_model=self.D)
        qa_head.eval()
        
        # 1. Test without context_start (no masking)
        start_logits, end_logits = qa_head(self.hidden_states)
        self.assertEqual(start_logits.shape, (self.B, self.T))
        self.assertEqual(end_logits.shape, (self.B, self.T))
        
        # Ensure no values are -inf
        self.assertFalse(torch.isinf(start_logits).any(), "Logits should not be inf/minus-inf when context_start is None")
        self.assertFalse(torch.isinf(end_logits).any(), "Logits should not be inf/minus-inf when context_start is None")

        # 2. Test with context_start (masking applied)
        context_start = 5
        start_logits_masked, end_logits_masked = qa_head(self.hidden_states, context_start=context_start)
        
        # Check shapes are still the same
        self.assertEqual(start_logits_masked.shape, (self.B, self.T))
        self.assertEqual(end_logits_masked.shape, (self.B, self.T))
        
        # Positions before context_start must be masked to -inf
        min_val = torch.finfo(start_logits_masked.dtype).min
        for b in range(self.B):
            for t in range(context_start):
                self.assertEqual(start_logits_masked[b, t].item(), min_val, f"start_logits at pos {t} should be masked to -inf")
                self.assertEqual(end_logits_masked[b, t].item(), min_val, f"end_logits at pos {t} should be masked to -inf")
        
        # Positions from context_start onwards should NOT be masked to -inf
        for b in range(self.B):
            for t in range(context_start, self.T):
                self.assertNotEqual(start_logits_masked[b, t].item(), min_val, f"start_logits at pos {t} should not be masked")
                self.assertNotEqual(end_logits_masked[b, t].item(), min_val, f"end_logits at pos {t} should not be masked")

if __name__ == "__main__":
    unittest.main()
