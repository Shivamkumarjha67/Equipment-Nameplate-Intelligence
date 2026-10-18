import unittest
from unittest.mock import MagicMock
from src.rag.query_engine import GraphQueryEngine
from src.graph.schema import SpecificationModel, EquipmentRecord


class TestQuerySafety(unittest.TestCase):
    def setUp(self):
        self.mock_connector = MagicMock()
        self.engine = GraphQueryEngine(
            connector=self.mock_connector, api_key="dummy-key", model="dummy-model"
        )

    def test_cypher_read_only_enforcement(self):
        """Verify that dangerous Cypher mutations are blocked before touching the database."""
        dangerous_queries = [
            "MATCH (e:Equipment) DELETE e",
            "MATCH (e:Equipment) SET e.model = 'HACKED'",
            "CREATE (m:Manufacturer {name: 'EVIL'})",
            "MERGE (l:PlantArea {code: 'NEW'})",
            "DROP CONSTRAINT equipment_serial",
        ]

        for q in dangerous_queries:
            with self.assertRaises(PermissionError):
                self.engine.execute_cypher(q)

    def test_valid_cypher_passes_to_driver(self):
        """Verify that read-only MATCH/RETURN queries pass through to the driver."""
        mock_session = MagicMock()
        self.mock_connector.driver.session.return_value.__enter__.return_value = mock_session
        mock_record = MagicMock()
        mock_record.data.return_value = {"serial": "SN-1234", "power": 5.5}
        mock_session.run.return_value = [mock_record]

        res = self.engine.execute_cypher("MATCH (e:Equipment) RETURN e.serial_number AS serial")

        self.assertEqual(len(res), 1)
        self.assertEqual(res[0]["serial"], "SN-1234")

    def test_pydantic_schema_validation(self):
        """Verify Pydantic models validate expected types."""
        spec = SpecificationModel(
            voltage_v=400,
            current_a=11.2,
            power_kw=5.5,
            frequency_hz=50,
            rpm=1440,
        )
        self.assertEqual(spec.voltage_v, 400)
        self.assertEqual(spec.power_kw, 5.5)

        record = EquipmentRecord(
            serial_number="SN-TEST-1234",
            model="IND-500",
            manufacturer="WEG",
            location_code="BAY-01",
            specifications=spec,
        )
        self.assertEqual(record.serial_number, "SN-TEST-1234")
        self.assertEqual(record.specifications.voltage_v, 400)


if __name__ == "__main__":
    unittest.main()

