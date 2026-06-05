# Agentic RAG project using Langgraph
from langgraph.graph import StateGraph
from ollama import embed, chat
import chromadb
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent.parent.parent
# Loading chromadb
chroma_client = chromadb.PersistentClient(
    path=(BASE_DIR / "rag_chatbot_project/chroma_db")

)
# print("BASE Directory: ", BASE_DIR)

collection = chroma_client.get_collection(
    name="rag_collection"
)

# print("Collection Count : ", collection.count())

question = input("Please ask your question  or enter /bye: ")
state = {
    "question": question,
    "context": [],
    "attempts": 0,
    "answer": ""
}


def planner(state):
    state["need_retrieval"] = True

    return state


def planner_router(state):
    if state["need_retrieval"]:
        return "retrieve"

    return "answer"


def retrieve(state):

    question = state["question"]
    response = embed(
        model="qwen3-embedding:0.6b",
        input=question
    )

    query_embedding = response["embeddings"][0]
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=10
    )
    # print("\nRetrieved Chunks:\n")

    # for i, chunk in enumerate(results["documents"][0]):
    #     print(f"\nChunk {i+1}")
    #     print(chunk[:500])

    state["attempts"] += 1

    state["context"] += results["documents"][0]

    return state


def evaluate_context(state):
    prompt = f"""
Question:
{state["question"]}

Context:
{state["context"]}

Determine whether the context is sufficient for generating the best answer.


If sufficient:

Reply ONLY:
YES

If insufficient:
Reply ONLY with a short retrieval query

Do not explain your reasoning.
"""
    response = chat(
        model="gemma4:e2b",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )
    decision = (
        response["message"]["content"].strip().upper()
    )
    print("Decision: ", decision)

    if "YES" in decision:
        state["enough_context"] = True

    else:
        state["enough_context"] = False
        state["question"] = decision

    return state


def context_router(state):
    if state["enough_context"]:
        return "answer"
    if state["attempts"] >= 3:
        return "answer"

    return "retrieve"


def answer(state):
    prompt = f"""
Question:
{state["question"]}

Context:
{state["context"]}

-Answer the question using context
-Answer in 5 bullet points
"""

    response = chat(
        model="gemma4:e2b",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]

    )
    state["answer"] = (
        response["message"]["content"]
    )

    return state


graph_builder = StateGraph(dict)

graph_builder.add_node(
    "planner",
    planner
)
graph_builder.add_node(
    "retrieve",
    retrieve
)

graph_builder.add_node(
    "answer",
    answer
)

graph_builder.add_node(
    "evaluate_context",
    evaluate_context
)


graph_builder.set_entry_point(
    "planner"
)
graph_builder.add_conditional_edges(
    "planner",
    planner_router
)

graph_builder.add_edge(
    "retrieve",
    "evaluate_context"
)

graph_builder.add_conditional_edges(
    "evaluate_context",
    context_router
)

graph_builder.set_finish_point(
    "answer"
)

graph = graph_builder.compile()

result = graph.invoke(state)
print(result["answer"])
