import torch
import unittest
from unittest.mock import MagicMock, patch
import sys
import os

# Add src to path to allow imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

# Mock unsloth and trl BEFORE importing rl_simple
for mod_name in ["unsloth", "trl", "peft"]:
    mock = MagicMock()
    mock.__spec__ = MagicMock()
    sys.modules[mod_name] = mock

from src.adversarial.rl_simple import CombinedReward

class MockBatchEncoding:
    def __init__(self, input_ids, attention_mask):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        
    def to(self, device):
        # Since we are on CPU, just return self
        # If we were mocking GPU, we might need to simulate moving tensors
        return self

class MockTokenizer:
    def __init__(self):
        self.vocab_size = 1000
        self.pad_token_id = 0
        
    def __len__(self):
        return self.vocab_size
    
    def __call__(self, texts, padding=True, truncation=True, max_length=512, return_tensors="pt"):
        # Simple mock tokenization: map unique words to IDs
        batch_ids = []
        max_len = 0
        for text in texts:
            # Simple hash-based tokenization for testing
            # Use deterministic hash for reproducibility
            tokens = [abs(hash(w)) % (self.vocab_size - 1) + 1 for w in text.split()]
            batch_ids.append(tokens)
            max_len = max(max_len, len(tokens))
        
        # Pad
        padded_ids = []
        attention_mask = []
        for tokens in batch_ids:
            pad_len = max_len - len(tokens)
            padded = tokens + [0] * pad_len
            mask = [1] * len(tokens) + [0] * pad_len
            padded_ids.append(padded)
            attention_mask.append(mask)
            
        return MockBatchEncoding(
            input_ids=torch.tensor(padded_ids),
            attention_mask=torch.tensor(attention_mask)
        )

class TestDiversityScore(unittest.TestCase):
    def setUp(self):
        # Mock the __init__ to avoid loading models
        # We need to patch where the class is defined
        with patch("src.adversarial.rl_simple.CombinedReward.__init__", return_value=None):
            self.reward = CombinedReward()
            # Manually set attributes that __init__ would have set
            self.reward.device = "cpu"
            self.reward.diversity_weight = 1.0
            self.reward.detector_tokenizer = MockTokenizer()
            
    @patch("src.adversarial.rl_simple.profiler")
    def test_identical_batch(self, mock_profiler):
        print("\n--- Test Identical Batch ---")
        # Identical messages should have high penalty (near 1.0 * weight)
        responses = ["hello world", "hello world", "hello world"]
        penalties = self.reward._compute_diversity_penalties(responses)
        
        print(f"Penalties: {penalties}")
        # Jaccard of identical sets is 1.0. 
        # Formula: (1.0 - 0.1) / 0.4 = 2.25 -> clamped to 1.0
        for p in penalties:
            self.assertAlmostEqual(p, 1.0, delta=0.01)

    @patch("src.adversarial.rl_simple.profiler")
    def test_unique_batch(self, mock_profiler):
        print("\n--- Test Unique Batch ---")
        # Completely unique messages should have 0 penalty
        responses = ["apple banana", "cherry date", "elderberry fig"]
        penalties = self.reward._compute_diversity_penalties(responses)
        
        print(f"Penalties: {penalties}")
        # Jaccard should be 0.
        # Formula: (0 - 0.1) / 0.4 < 0 -> clamped to 0.0
        for p in penalties:
            self.assertEqual(p, 0.0)

    @patch("src.adversarial.rl_simple.profiler")
    def test_partial_overlap(self, mock_profiler):
        print("\n--- Test Partial Overlap ---")
        # "hello world" vs "hello mars" -> 1 common token (hello), 3 total unique (hello, world, mars)
        # Jaccard = 1/3 = 0.333...
        # Avg sim for each = 0.333...
        # Penalty = (0.333 - 0.1) / 0.4 = 0.233 / 0.4 = 0.583
        responses = ["hello world", "hello mars"]
        penalties = self.reward._compute_diversity_penalties(responses)
        
        print(f"Penalties: {penalties}")
        for p in penalties:
            self.assertGreater(p, 0.5)
            self.assertLess(p, 0.7)
            
    @patch("src.adversarial.rl_simple.profiler")
    def test_single_item(self, mock_profiler):
        print("\n--- Test Single Item ---")
        responses = ["hello world"]
        penalties = self.reward._compute_diversity_penalties(responses)
        print(f"Penalties: {penalties}")
        self.assertEqual(penalties, [0.0])

    @patch("src.adversarial.rl_simple.profiler")
    def test_out_of_bounds_tokens(self, mock_profiler):
        print("\n--- Test Out of Bounds Tokens ---")
        # Mock tokenizer to return an ID that is out of bounds
        # vocab_size is 100 (from MockTokenizer)
        # We'll manually inject a large ID
        
        # Create a custom mock for this test
        with patch.object(self.reward, "detector_tokenizer") as mock_tok:
            mock_tok.__len__.return_value = 100
            mock_tok.vocab_size = 100
            
            # Return a tensor with an out-of-bounds ID (e.g. 150)
            # and a valid ID (e.g. 50)
            input_ids = torch.tensor([[50, 150], [50, 20]])
            attention_mask = torch.tensor([[1, 1], [1, 1]])
            
            mock_tok.return_value = MockBatchEncoding(input_ids, attention_mask)
            mock_tok.return_value.to = lambda x: mock_tok.return_value # Mock .to()
            
            # This should NOT raise an index error because of the clamp/filter
            responses = ["dummy", "dummy"]
            try:
                penalties = self.reward._compute_diversity_penalties(responses)
                print(f"Penalties: {penalties}")
                # Both have token 50. Token 150 should be ignored.
                # So first msg has {50}, second has {50, 20}.
                # Intersection = {50} (size 1)
                # Union = {50, 20} (size 2)
                # Jaccard = 0.5
                # Penalty = (0.5 - 0.1) / 0.4 = 1.0
                self.assertAlmostEqual(penalties[0], 1.0, delta=0.01)
            except IndexError:
                self.fail("IndexError raised! The safety clamp did not work.")

if __name__ == "__main__":
    unittest.main()
