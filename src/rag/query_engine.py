from __future__ import annotations

import re
from google import genai
from google.genai import types
from src.config import GEMINI_API_KEY, GEMINI_MODEL
from src.graph.connection import Neo4jConnector


class GraphQueryEngine:

  def __init__(
      self,
      connector: Neo4jConnector,
      api_key: str = GEMINI_API_KEY,
      model: str = GEMINI_MODEL,
  ):
    self.connector = connector
    self.model = model
    self.client = genai.Client(api_key=api_key) if api_key else None

    self.system_prompt = """
You are an expert Neo4j Cypher translator for an industrial asset intelligence system.
Your job is to convert natural language maintenance questions into valid, read-only Cypher queries.

The database schema is:
- Nodes:
  - (:Equipment {serial_number: STRING, model: STRING, last_inspected: DATETIME})
  - (:Manufacturer {name: STRING})
  - (:PlantArea {code: STRING})
  - (:Specification {spec_id: STRING, voltage_v: INTEGER, current_a: FLOAT, power_kw: FLOAT, frequency_hz: INTEGER, rpm: INTEGER})
  - (:MaintenanceEvent {timestamp: DATETIME, note: STRING})

- Relationships:
  - (:Equipment)-[:MANUFACTURED_BY]->(:Manufacturer)
  - (:Equipment)-[:LOCATED_AT]->(:PlantArea)
  - (:Equipment)-[:HAS_SPEC]->(:Specification)
  - (:Equipment)-[:HAS_MAINTENANCE]->(:MaintenanceEvent)

CRITICAL RULES:
1. Return ONLY the raw Cypher query string starting with MATCH.
2. DO NOT wrap the output in markdown code fences or backticks (e.g., no ```cypher).
3. READ-ONLY: Never use CREATE, MERGE, SET, DELETE, or DROP.
4. For text filters, use toLower() for safety (e.g. toLower(m.name) CONTAINS 'siemens' or toLower(l.code) = toLower('BAY-01')).
5. Return informative column aliases: e.g. RETURN e.serial_number AS serial, e.model AS model, m.name AS manufacturer, l.code AS location, s.power_kw AS power, s.voltage_v AS voltage.
"""

  def generate_cypher(self, user_question: str) -> str:
    """Uses Gemini to translate natural language into a Cypher query."""
    if not self.client:
      raise ValueError(
          "Gemini API key is missing. Set GEMINI_API_KEY in src/config.py."
      )

    prompt = (
        f"{self.system_prompt}\n\nUser Question: {user_question.strip()}\nCypher"
        " Query:"
    )

    response = self.client.models.generate_content(
        model=self.model,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.0,
        ),
    )

    raw_text = response.text.strip()
    # Strip any accidental markdown formatting
    cleaned_cypher = re.sub(
        r"^```(?:cypher)?\s*", "", raw_text, flags=re.IGNORECASE
    )
    cleaned_cypher = re.sub(r"\s*```$", "", cleaned_cypher)
    return cleaned_cypher.strip()

  def execute_cypher(self, cypher_query: str) -> list[dict]:
    """Executes the query safely against Neo4j AuraDB."""
    forbidden = ["DELETE", "CREATE", "SET", "MERGE", "DROP", "REMOVE"]
    if any(cmd in cypher_query.upper().split() for cmd in forbidden):
      raise PermissionError(
          f"Query rejected: write/mutation command detected ({cypher_query})"
      )

    with self.connector.driver.session() as session:
      results = session.run(cypher_query)
      return [record.data() for record in results]

  def summarize_answer(
      self, user_question: str, cypher_query: str, records: list[dict]
  ) -> str:
    """Generates a concise engineering answer from the retrieved graph records."""
    if not records:
      return "No matching equipment or specifications found in the plant database."

    prompt = f"""
You are an industrial plant assistant. Answer the user's question clearly based strictly on the retrieved database records.

User Question: {user_question}
Cypher Executed: {cypher_query}
Database Records: {records}

State the findings concisely in 2-3 sentences. Mention the asset serial numbers, ratings, and locations where relevant.
"""
    response = self.client.models.generate_content(
        model=self.model,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.2,
        ),
    )
    return response.text.strip()

  def answer_question(self, user_question: str) -> dict:
    """Full Graph-RAG pipeline coordinator."""
    cypher = self.generate_cypher(user_question)
    data = self.execute_cypher(cypher)
    summary = self.summarize_answer(user_question, cypher, data)
    return {
        "cypher_query": cypher,
        "raw_records": data,
        "summary": summary,
    }