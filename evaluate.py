import json
import os
from pathlib import Path
from typing import Dict, List

from langsmith import Client
from langsmith.evaluation import evaluate

from app import run_agent


DATASET_NAME = "faq-support-agent"


DATASET_EXAMPLES = [
    {
        "question": "What is your refund policy?",
        "expected_answer": "Refunds are available within 30 days for unused items.",
    },
    {
        "question": "How long does shipping take?",
        "expected_answer": "Standard shipping takes 3-5 business days.",
    },
    {
        "question": "What warranty is included?",
        "expected_answer": "Products include a 1-year limited warranty.",
    },
    {
        "question": "Can I return an item?",
        "expected_answer": "Returns are accepted if the item is unused and in original packaging.",
    },
    {
        "question": "How do I reset my password?",
        "expected_answer": "You can reset your password from the sign-in page.",
    },
]


def build_local_results() -> List[Dict]:
    results = []
    for row in DATASET_EXAMPLES:
        result = run_agent(row["question"])
        prediction = result["answer"]
        expected = row["expected_answer"]
        score = int(expected.lower() in prediction.lower())
        results.append(
            {
                "question": row["question"],
                "prediction": prediction,
                "expected": expected,
                "score": score,
            }
        )
    return results


def run_langsmith_evaluation() -> dict:
    if not os.environ.get("LANGSMITH_API_KEY"):
        return {"status": "skipped", "reason": "LANGSMITH_API_KEY is not set"}

    client = Client()

    try:
        client.read_dataset(dataset_name=DATASET_NAME)
    except Exception:
        dataset = client.create_dataset(
            dataset_name=DATASET_NAME,
            description="Synthetic evaluation dataset for a simple LangGraph FAQ agent.",
        )
        for row in DATASET_EXAMPLES:
            client.create_example(
                dataset_id=dataset.id,
                inputs={"question": row["question"]},
                outputs={"answer": row["expected_answer"]},
            )

    def predict(inputs: dict) -> dict:
        result = run_agent(inputs["question"])
        return {"output": result["answer"]}

    def correctness(run, example):
        prediction = str(run.outputs.get("output", "")).strip().lower()
        expected = str(example.outputs.get("answer", "")).strip().lower()
        score = int(expected in prediction or prediction in expected)
        return {"score": score, "comment": f"Expected: {expected}"}

    results = evaluate(
        predict,
        data=DATASET_NAME,
        evaluators=[correctness],
        experiment_prefix="faq-support-agent",
        description="Evaluate a simple LangGraph FAQ agent with LangSmith",
        metadata={"agent": "langgraph-faq", "workflow": "state-graph"},
        client=client,
        blocking=True,
    )
    return {"status": "completed", "experiment_name": results.experiment_name}


def main() -> None:
    local_results = build_local_results()
    output_path = Path("results/local_eval.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(local_results, indent=2))

    print("Local evaluation results saved to:", output_path)
    for item in local_results:
        print(f"- {item['question']}: score={item['score']}")

    langsmith_result = run_langsmith_evaluation()
    print("LangSmith status:", langsmith_result)


if __name__ == "__main__":
    main()
