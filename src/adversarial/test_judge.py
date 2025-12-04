import unittest
from unittest.mock import MagicMock, patch
import torch
import sys
import os

# Mock dependencies to avoid import errors or heavy loading
sys.modules["unsloth"] = MagicMock()
sys.modules["unsloth"].__spec__ = MagicMock()

sys.modules["trl"] = MagicMock()
sys.modules["trl"].__spec__ = MagicMock()

sys.modules["peft"] = MagicMock()
sys.modules["peft"].__spec__ = MagicMock()

sys.modules["wandb"] = MagicMock()
sys.modules["wandb"].__spec__ = MagicMock()

# Mock internal dependencies if needed
# We need to make sure we can import rl_simple
# rl_simple imports BypassLogger from .bypass_logger
# We might need to ensure that import works or is mocked

# Add src to path to allow imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.adversarial.rl_simple import CombinedReward, parse_judge_score

class TestJudgeScoring(unittest.TestCase):
    def setUp(self):
        # Mock CombinedReward
        # We bypass __init__ to avoid loading models
        with patch("src.adversarial.rl_simple.CombinedReward.__init__", return_value=None):
            self.reward = CombinedReward("dummy_path")
        
        self.reward.device = "cpu"
        self.reward.use_judge = True
        self.reward.judge = MagicMock()
        self.reward.judge_tokenizer = MagicMock()
        # Mock profiler since it's used in _get_judge_scores
        self.reward.profiler = MagicMock()
        # Patch the global profiler used in rl_simple if it exists, 
        # but looking at code it seems 'profiler' might be a global in rl_simple.py
        # Let's check if we need to patch it.
        # In rl_simple.py: `profiler = Profiler()` is at module level.
        # We should patch it.
        
    @patch("src.adversarial.rl_simple.profiler")
    def test_judge_scoring_slicing(self, mock_profiler):
        """Test that input prompt is correctly sliced off from judge output."""
        
        # Setup
        prompts = ["Test SMS"]
        
        # Mock tokenizer to return inputs of length 5
        # input_ids shape: [1, 5]
        input_ids = torch.tensor([[1, 1, 1, 1, 1]]) 
        mock_inputs = MagicMock()
        mock_inputs.input_ids = input_ids
        mock_inputs.to.return_value = mock_inputs
        
        # Mock apply_chat_template
        self.reward.judge_tokenizer.apply_chat_template.return_value = "Prompt"
        
        # Mock tokenizer call
        self.reward.judge_tokenizer.return_value = mock_inputs
        self.reward.judge_tokenizer.pad_token_id = 0
        
        # Mock generate output
        # It should return [input_ids + generated_ids]
        # Let's say generated ids are [999, 999] (length 2)
        # Total shape: [1, 7]
        generated_ids = torch.tensor([[999, 999]])
        full_output = torch.cat([input_ids, generated_ids], dim=1)
        
        self.reward.judge.generate.return_value = full_output
        
        # Mock decode
        # We want to verify that decode is called with ONLY generated_ids
        def side_effect_decode(ids, skip_special_tokens=True):
            if torch.equal(ids, generated_ids[0]):
                return '{"score": 8, "reason": "test"}'
            elif torch.equal(ids, full_output[0]):
                return 'Prompt... {"score": 8, "reason": "test"}' # This would be the bug case
            else:
                return "Unknown"
                
        self.reward.judge_tokenizer.decode.side_effect = side_effect_decode
        
        # Run
        scores = self.reward._get_judge_scores(prompts)
        
        # Verify
        self.assertEqual(scores, [0.8])
        
        # Verify decode was called with the correct slice
        # The slice should be from input_ids.shape[1] (which is 5) to end
        # So it should be full_output[0][5:] which is [999, 999]
        
        # We can also check the call args of decode
        call_args = self.reward.judge_tokenizer.decode.call_args
        self.assertTrue(torch.equal(call_args[0][0], generated_ids[0]), 
                        "decode was not called with correctly sliced tokens")

    def test_parse_judge_score(self):
        """Test score parsing logic."""
        # JSON format
        self.assertEqual(parse_judge_score('{"score": 8, "reason": "good"}'), 0.8)
        self.assertEqual(parse_judge_score('{"score": 8}'), 0.8)
        self.assertEqual(parse_judge_score('Some text {"score": 7, "reason": "ok"} end'), 0.7)
        self.assertEqual(parse_judge_score('{"score": 0, "reason": "bad"}'), 0.0)
        
        # Fallback formats
        self.assertEqual(parse_judge_score("Score: 8"), 0.8)
        self.assertEqual(parse_judge_score("8/10"), 0.8)
        self.assertEqual(parse_judge_score("I rate this 0/10"), 0.0)
        self.assertEqual(parse_judge_score("Score: 10"), 1.0)
        self.assertEqual(parse_judge_score("Garbage"), 0.5) # Default
        self.assertEqual(parse_judge_score("7"), 0.7)
        self.assertEqual(parse_judge_score("Score: 9.5"), 0.95)
        self.assertEqual(parse_judge_score("<think>...</think> 5"), 0.5)

if __name__ == "__main__":
    unittest.main()
