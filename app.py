from typing import TypedDict

from langgraph.graph import END, START, StateGraph


class AgentState(TypedDict):
    question: str
    answer: str
    source: str
    confidence: str


FAQ_KNOWLEDGE = {
    "refund": "Refunds are available within 30 days for unused items.",
    "shipping": "Standard shipping takes 3-5 business days.",
    "warranty": "Products include a 1-year limited warranty.",
    "returns": "Returns are accepted if the item is unused and in original packaging.",
    "account": "You can reset your password from the sign-in page.",
}


def retrieve_answer(state: AgentState) -> AgentState:
    question = state["question"].strip().lower()
    for keyword, answer in FAQ_KNOWLEDGE.items():
        if keyword in question:
            return {
                "question": state["question"],
                "answer": answer,
                "source": keyword,
                "confidence": "high",
            }

    fallback_answer = (
        "I could not find a precise answer. Please contact support for a human assist."
    )
    return {
        "question": state["question"],
        "answer": fallback_answer,
        "source": "fallback",
        "confidence": "low",
    }


def format_response(state: AgentState) -> AgentState:
    answer = state["answer"]
    if state["source"] == "fallback":
        answer = f"{answer}"
    return {
        "question": state["question"],
        "answer": answer,
        "source": state["source"],
        "confidence": state["confidence"],
    }


workflow = StateGraph(AgentState)
workflow.add_node("retrieve", retrieve_answer)
workflow.add_node("finalize", format_response)
workflow.add_edge(START, "retrieve")
workflow.add_edge("retrieve", "finalize")
workflow.add_edge("finalize", END)

app = workflow.compile()


def run_agent(question: str) -> dict:
    return app.invoke({"question": question, "answer": "", "source": "", "confidence": ""})


if __name__ == "__main__":
    sample = "What is your refund policy?"
    result = run_agent(sample)
    print(f"Question: {sample}")
    print(f"Answer: {result['answer']}")
    print(f"Source: {result['source']}")
    print(f"Confidence: {result['confidence']}")
