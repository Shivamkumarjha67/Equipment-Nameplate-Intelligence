from __future__ import annotations
from neo4j import GraphDatabase
from src.config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD

class Neo4jConnector:
    def __init__(self, uri: str = NEO4J_URI, user: str = NEO4J_USER, password: str = NEO4J_PASSWORD):
        self.uri = uri
        self.user = user
        self.password = password
        self.driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))

    def close(self):
        if self.driver:
            self.driver.close()

    def verify_connection(self) -> bool:
        try:
            self.driver.verify_connectivity()
            return True
        except Exception as e:
            print(f"[!] Neo4j connection failed: {e}")
            return False

    def init_schema(self):
        """Applies uniqueness constraints and indexes for industrial assets."""
        queries = [
            "CREATE CONSTRAINT equipment_serial IF NOT EXISTS FOR (e:Equipment) REQUIRE e.serial_number IS UNIQUE",
            "CREATE CONSTRAINT manufacturer_name IF NOT EXISTS FOR (m:Manufacturer) REQUIRE m.name IS UNIQUE",
            "CREATE CONSTRAINT location_code IF NOT EXISTS FOR (l:PlantArea) REQUIRE l.code IS UNIQUE"
        ]
        with self.driver.session() as session:
            for q in queries:
                session.run(q)
        print("[OK] Neo4j AuraDB constraints initialized successfully.")