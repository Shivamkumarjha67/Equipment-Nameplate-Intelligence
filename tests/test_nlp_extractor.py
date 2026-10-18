import unittest
from src.ocr.extractor import NameplateExtractor


class DummyExtractor(NameplateExtractor):
    """Subclass that bypasses heavy EasyOCR Reader initialization for pure parsing tests."""
    def __init__(self):
        pass


class TestNLPExtractor(unittest.TestCase):
    def setUp(self):
        self.extractor = DummyExtractor()

    def test_manufacturer_recognition(self):
        tokens = [
            {"text": "SIEMENS AKTIENGESELLSCHAFT", "confidence": 0.95},
            {"text": "MADE IN GERMANY", "confidence": 0.90},
        ]
        parsed = self.extractor.parse_fields(tokens)
        self.assertEqual(parsed["manufacturer"]["value"], "SIEMENS")
        self.assertGreaterEqual(parsed["manufacturer"]["confidence"], 0.95)

    def test_voltage_and_frequency_disambiguation(self):
        # Simulates OCR confusion: 'SO HZ' for 50 HZ, '4OO V' with letter O instead of zero
        tokens = [
            {"text": "VOLTS: 4OO V", "confidence": 0.88},
            {"text": "FREQ: SO HZ", "confidence": 0.85},
        ]
        parsed = self.extractor.parse_fields(tokens)
        self.assertEqual(parsed["voltage"]["value"], 400)
        self.assertEqual(parsed["frequency"]["value"], 50)

    def test_serial_number_format(self):
        tokens = [
            {"text": "SER NO: SN-K789-M234", "confidence": 0.91},
        ]
        parsed = self.extractor.parse_fields(tokens)
        self.assertEqual(parsed["serial_number"]["value"], "SN-K789-M234")

    def test_floating_point_current_and_power(self):
        tokens = [
            {"text": "AMPS: 14.8 A", "confidence": 0.92},
            {"text": "RATING: 7.5 KW", "confidence": 0.93},
        ]
        parsed = self.extractor.parse_fields(tokens)
        self.assertEqual(parsed["current"]["value"], 14.8)
        self.assertEqual(parsed["power"]["value"], 7.5)


if __name__ == "__main__":
    unittest.main()

