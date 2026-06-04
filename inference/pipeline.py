from typing import List, Dict, Any

class ThaiNLPPipeline:
    """
    High-level inference class that coordinates text preprocessing, 
    backbone encoder pass, and head-specific forward passes to produce predictions.
    """
    def __init__(self, model_dir: str, device: str = "cpu"):
        self.device = device
        # TODO: Load model, tokenizer, and config

    def predict(self, text: str, tasks: List[str]) -> Dict[str, Any]:
        """
        Runs prediction for the specified tasks on the input text.
        Supported tasks: 'ner', 'sentiment', 'qa'
        """
        # TODO: Process text, run through model and task heads, format output
        return {}
