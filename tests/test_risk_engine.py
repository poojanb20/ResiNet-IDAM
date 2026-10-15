
import unittest

from src.zero_trust.risk_engine import RiskInput, calculate_risk


class TestRiskEngine(unittest.TestCase):

    def test_low_risk(self):
        result = calculate_risk(RiskInput(0.05, 0.10, 0.20, 0.95))
        self.assertEqual(result["risk_level"], "LOW")
        self.assertEqual(result["recommended_action"], "ALLOW_AND_MONITOR")

    def test_high_risk(self):
        result = calculate_risk(RiskInput(0.95, 0.95, 1.0, 0.10))
        self.assertEqual(result["risk_level"], "HIGH")
        self.assertEqual(result["recommended_action"], "RESTRICT_AND_ALERT")

    def test_invalid_input(self):
        with self.assertRaises(ValueError):
            calculate_risk(RiskInput(1.2, 0.5, 0.5, 0.5))


if __name__ == "__main__":
    unittest.main()
