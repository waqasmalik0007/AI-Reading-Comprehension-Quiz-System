"""
test_inference.py — Unit tests for the inference engine.

Tests:
  - Model loading
  - Answer verification
  - Distractor generation
  - Hint generation
  - Random sample loading
  - Full inference pipeline
  - Latency check (<10s per request)
"""

import os
import sys
import time
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


class TestInferenceEngine(unittest.TestCase):
    """Test suite for the InferenceEngine class."""

    @classmethod
    def setUpClass(cls):
        """Load the engine once for all tests."""
        from src.inference import InferenceEngine
        cls.engine = InferenceEngine()

    def test_models_loaded(self):
        """Models should be loaded successfully."""
        self.assertTrue(self.engine.models_loaded, "Models failed to load")

    def test_random_sample(self):
        """Should return a valid sample dict."""
        sample = self.engine.get_random_sample()
        self.assertIsNotNone(sample)
        self.assertIn('article', sample)
        self.assertIn('question', sample)
        self.assertIn('options', sample)
        self.assertIn('correct_answer', sample)
        self.assertIn(sample['correct_answer'], ['A', 'B', 'C', 'D'])

    def test_verify_answer(self):
        """Verify answer should return a probability between 0 and 1."""
        sample = self.engine.get_random_sample()
        prob, feat, latency = self.engine.verify_answer(
            sample['article'], sample['question'],
            sample['options'][sample['correct_answer']]
        )
        self.assertGreaterEqual(prob, 0.0)
        self.assertLessEqual(prob, 1.0)
        self.assertIsInstance(feat, dict)
        self.assertGreater(len(feat), 0)
        self.assertGreater(latency, 0)

    def test_rank_all_options(self):
        """Ranking should return 4 options sorted by probability."""
        sample = self.engine.get_random_sample()
        ranked, latency = self.engine.rank_all_options(
            sample['article'], sample['question'], sample['options']
        )
        self.assertEqual(len(ranked), 4)
        # Check sorted descending
        probs = [r[2] for r in ranked]
        self.assertEqual(probs, sorted(probs, reverse=True))

    def test_generate_distractors(self):
        """Should generate exactly 3 distractors."""
        sample = self.engine.get_random_sample()
        correct_text = sample['options'][sample['correct_answer']]
        distractors, latency = self.engine.generate_distractors(
            sample['article'], sample['question'], correct_text
        )
        self.assertEqual(len(distractors), 3)
        # Correct answer should NOT be among distractors
        for d in distractors:
            self.assertNotEqual(d.strip().lower(), correct_text.strip().lower())

    def test_generate_hints(self):
        """Should generate 1-3 hints."""
        sample = self.engine.get_random_sample()
        hints, latency = self.engine.generate_hints(
            sample['article'], sample['question'], n_hints=3
        )
        self.assertGreater(len(hints), 0)
        self.assertLessEqual(len(hints), 3)
        for hint in hints:
            self.assertIsInstance(hint, str)
            self.assertGreater(len(hint), 0)

    def test_full_inference(self):
        """Full inference pipeline should produce all required fields."""
        sample = self.engine.get_random_sample()
        result = self.engine.full_inference(
            sample['article'],
            question=sample['question'],
            options=sample['options']
        )
        self.assertIn('ranked_options', result)
        self.assertIn('predicted_answer', result)
        self.assertIn('hints', result)
        self.assertIn('latencies', result)
        self.assertIn(result['predicted_answer'], ['A', 'B', 'C', 'D'])

    def test_latency_under_10s(self):
        """Single inference must complete in under 10 seconds (project requirement)."""
        sample = self.engine.get_random_sample()
        start = time.time()
        result = self.engine.full_inference(
            sample['article'],
            question=sample['question'],
            options=sample['options']
        )
        elapsed = time.time() - start
        self.assertLess(elapsed, 10.0, f"Inference took {elapsed:.2f}s, exceeds 10s limit")

    def test_feature_count(self):
        """Verification features should have 11 dimensions."""
        sample = self.engine.get_random_sample()
        _, feat, _ = self.engine.verify_answer(
            sample['article'], sample['question'],
            sample['options']['A']
        )
        self.assertEqual(len(feat), 11)


if __name__ == '__main__':
    unittest.main(verbosity=2)
