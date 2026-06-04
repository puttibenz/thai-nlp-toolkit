from typing import List, Dict, Any

def compute_ner_metrics(predictions: List[List[str]], references: List[List[str]]) -> Dict[str, float]:
    """Computes precision, recall, and F1 score for NER predictions."""
    # TODO: Implement sequence labeling F1 computation
    return {"ner_f1": 0.0}

def compute_sentiment_metrics(predictions: List[int], references: List[int]) -> Dict[str, float]:
    """Computes accuracy and macro F1 for Sentiment predictions."""
    # TODO: Implement classification metrics
    return {"sentiment_accuracy": 0.0, "sentiment_macro_f1": 0.0}

def compute_qa_metrics(predictions: List[Dict[str, Any]], references: List[Dict[str, Any]]) -> Dict[str, float]:
    """Computes exact match (EM) and token-level F1 for QA predictions."""
    # TODO: Implement span extraction EM and F1 evaluation
    return {"qa_exact_match": 0.0, "qa_f1": 0.0}
