import unittest
from src.ml.feature_extractor import NameplateFeatureExtractor


class TestPhysicsValidation(unittest.TestCase):
    def setUp(self):
        self.extractor = NameplateFeatureExtractor()

    def test_plausible_three_phase_power(self):
        """Verify that a physically consistent electrical triplet has low discrepancy."""
        parsed_fields = {
            "voltage": {"value": 400, "confidence": 0.95},
            "current": {"value": 10.5, "confidence": 0.92},
            "power": {"value": 5.5, "confidence": 0.93},
            "frequency": {"value": 50, "confidence": 0.95},
            "manufacturer": {"value": "SIEMENS", "confidence": 0.99},
            "serial_number": {"value": "SN-ABCD-1234", "confidence": 0.92},
        }
        prep_res = {"blur_score": 150.0, "corners_detected": True}
        tokens = [{"confidence": 0.92, "text": "400V"}]

        features = self.extractor.extract_features(prep_res, tokens, parsed_fields)
        self.assertLess(features["power_discrepancy"], 0.20)
        self.assertEqual(features["completeness_ratio"], 1.0)

    def test_implausible_power_discrepancy_penalty(self):
        """Verify that an impossible electrical combination gets flagged."""
        parsed_fields = {
            "voltage": {"value": 400, "confidence": 0.95},
            "current": {"value": 10.5, "confidence": 0.92},
            "power": {"value": 55.0, "confidence": 0.93},  # Glare erased decimal: 5.5 kW -> 55 kW
            "frequency": {"value": 50, "confidence": 0.95},
            "manufacturer": {"value": "ABB", "confidence": 0.99},
            "serial_number": {"value": "SN-XYZW-5678", "confidence": 0.92},
        }
        prep_res = {"blur_score": 120.0, "corners_detected": True}
        tokens = [{"confidence": 0.88, "text": "55KW"}]

        features = self.extractor.extract_features(prep_res, tokens, parsed_fields)
        self.assertGreater(features["power_discrepancy"], 5.0)

    def test_missing_fields_penalty(self):
        """Verify missing critical fields apply maximum discrepancy penalty."""
        parsed_fields = {
            "voltage": {"value": None, "confidence": 0.0},
            "current": {"value": None, "confidence": 0.0},
            "power": {"value": 5.5, "confidence": 0.93},
        }
        prep_res = {"blur_score": 50.0, "corners_detected": False}
        tokens = []

        features = self.extractor.extract_features(prep_res, tokens, parsed_fields)
        self.assertEqual(features["power_discrepancy"], 5.0)
        self.assertLess(features["completeness_ratio"], 0.5)


if __name__ == "__main__":
    unittest.main()

