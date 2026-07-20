# LangGraph + LangSmith demo

This project builds a tiny LangGraph-based FAQ agent and wires it to a LangSmith evaluation flow.

## What it contains

- A simple stateful workflow in `app.py`
- A synthetic evaluation dataset in `evaluate.py`
- A local JSON export of evaluation results under `results/`

## Run locally

```bash
cd ~/langgraph-langsmith-demo
source .venv/bin/activate
python app.py
python evaluate.py
```

## LangSmith setup

To run the SDK-backed experiment in LangSmith, set:

```bash
export LANGSMITH_API_KEY="your-key"
export LANGSMITH_TRACING="true"
```

Then run:

```bash
python evaluate.py
```

The script will:

1. Run the agent against a synthetic FAQ dataset.
2. Save a local JSON evaluation file.
3. Send the evaluation to LangSmith if the API key is configured.

## Notes

This example intentionally stays simple so the LangGraph state model and LangSmith evaluation loop are easy to follow.
