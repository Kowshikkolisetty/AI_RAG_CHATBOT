import argparse
import json
import os

import requests

SAMPLE_QUERIES = [
    "What is Agentic AI according to the eBook?",
    "How do AI agents differ from traditional automation systems?",
    "What are the core components of an Agentic Architecture?",
    "What role does memory play in Agentic AI workflows?",
    "Who won the 2022 FIFA World Cup?",
]


def run_query_batch(base_url: str = "http://localhost:8000"):
    results = []
    for question in SAMPLE_QUERIES:
        response = requests.post(f"{base_url}/chat", json={"query": question}, timeout=120)
        response.raise_for_status()
        payload = response.json()
        results.append(
            {
                "question": question,
                "answer": payload.get("final_answer"),
                "retrieved_context": payload.get("retrieved_context", []),
                "confidence_score": payload.get("confidence_score", 0.0),
            }
        )
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the sample Agentic AI benchmark queries against the local API.")
    parser.add_argument("--base-url", default=os.getenv("API_BASE_URL", "http://localhost:8000"))
    args = parser.parse_args()

    for result in run_query_batch(args.base_url):
        print(json.dumps(result, indent=2))
        print("\n---\n")
