from __future__ import annotations
from src.graph.connection import Neo4jConnector

class EquipmentGraphIngestor:
    def __init__(self, connector: Neo4jConnector):
        self.connector = connector

    def ingest_nameplate(
        self,
        parsed_record: dict,
        location_code: str = "BAY-01",
        maintenance_note: str | None = None,
    ) -> bool:
        serial = parsed_record.get("serial_number", {}).get("value")
        mfr = parsed_record.get("manufacturer", {}).get("value")

        if not serial or not mfr:
            return False

        cypher_query = """
        // 1. Upsert Equipment
        MERGE (e:Equipment {serial_number: $serial})
        SET e.model = $model,
            e.last_inspected = datetime()

        // 2. Upsert Manufacturer & Relationship
        MERGE (m:Manufacturer {name: $manufacturer})
        MERGE (e)-[:MANUFACTURED_BY]->(m)

        // 3. Upsert Plant Location
        MERGE (l:PlantArea {code: $location})
        MERGE (e)-[:LOCATED_AT]->(l)

        // 4. Upsert Technical Specifications
        MERGE (s:Specification {spec_id: "SPEC-" + $serial})
        SET s.voltage_v = $voltage,
            s.current_a = $current,
            s.power_kw = $power,
            s.frequency_hz = $frequency,
            s.rpm = $rpm
        MERGE (e)-[:HAS_SPEC]->(s)
        """

        params = {
            "serial": serial,
            "model": parsed_record.get("model", {}).get("value") or "UNKNOWN",
            "manufacturer": mfr,
            "location": location_code,
            "voltage": parsed_record.get("voltage", {}).get("value"),
            "current": parsed_record.get("current", {}).get("value"),
            "power": parsed_record.get("power", {}).get("value"),
            "frequency": parsed_record.get("frequency", {}).get("value"),
            "rpm": parsed_record.get("rpm", {}).get("value"),
        }

        with self.connector.driver.session() as session:
            session.run(cypher_query, params)

            if maintenance_note:
                maint_query = """
                MATCH (e:Equipment {serial_number: $serial})
                CREATE (ev:MaintenanceEvent {
                    timestamp: datetime(),
                    note: $note
                })
                CREATE (e)-[:HAS_MAINTENANCE]->(ev)
                """
                session.run(maint_query, {"serial": serial, "note": maintenance_note})

        return True