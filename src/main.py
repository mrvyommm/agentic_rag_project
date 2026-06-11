# Agentic RAG project using Langgraph


from langgraph.graph import StateGraph
from ollama import embed, chat
import chromadb
from pathlib import Path
from dotenv import load_dotenv
import os
from tavily import TavilyClient
import json

BASE_DIR = Path(__file__).resolve().parent.parent.parent

load_dotenv()
tavily_client = TavilyClient(
    api_key=os.getenv("TAVILY_API_KEY")
)

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
    "tool": "",
    "context": [],
    "attempts": 0,
    "answer": ""
}


def planner(state):
    prompt = f"""

You are a routing agent.

Question:
{state["question"]}

Tools Available:
Available Tools:

rag
Use for:
- AI Engineering
- Foundation Models
- Embeddings
- Prompt Engineering
- AI Stack concepts

calculator
Use for:
- Mathematical calculations

direct_llm
Use for:
- General knowledge
- Questions outside the knowledge base

web_search
Use for:
-Current events
-Latest knowledge
-Recent news
-Information after model training

-Choose the best tool | Reply with valid JSON ONLY:

Example:

{{"tool":"calculator"}}

{{"tool":"rag"}}

{{"tool":"direct_llm"}}

{{"tool":"web_search"}}

-Do not explain.
-Do not provide reasoning.
-do not output multiple options.

    """
    decision = call_llm(prompt)
    print("Decision: ", decision)

    parsed = json.loads(decision)

    # state["tool"] = parsed["tool"]

    if "calculator" in decision.lower():
        state["tool"] = "calculator"

    elif "rag" in decision.lower():
        state["tool"] = "rag"

    elif "web_search" in decision.lower():
        state["tool"] = "web_search"

    else:
        state["tool"] = "direct_llm"

    return state


def planner_router(state):

    if state["tool"] == "calculator":
        return "call_calculator"

    elif state["tool"] == "rag":
        return "retrieve"

    elif state["tool"] == "web_search":
        return "call_web_search"

    else:
        return "web_search"


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


def call_calculator(state):
    expression = state["question"]

    try:
        result = eval(expression)
        state["answer"] = str(result)

    except Exception:
        state["answer"] = "Invalid Expression"

    return state


def call_direct_llm(state):
    prompt = f"""
Question:
{state["question"]} 

You are a genious mentor:
-Answer the question in five bullet points.
-Keep it precise and clear.

"""

    state["answer"] = call_llm(prompt)

    return state


def call_web_search(state):
    query = state["question"]

    results = tavily_client.search(
        query=query,
        max_results=5
    )

    context = ""

    for item in results["results"]:

        context += item["title"]

        context += "\n"

        context += item["content"]
        context += "\n\n"

    prompt = f"""
Question:
{query}

Search Results:
{context}

Take the search results and answer in 5 bullet points.

"""

    state["answer"] = call_llm(prompt)

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

    decision = call_llm(prompt).strip().upper()

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

    state["answer"] = call_llm(prompt)

    return state


def call_llm(prompt):
    response = chat(
        model="gemma4:e2b",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )
    answer = response["message"]["content"]

    return answer


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
    "call_calculator",
    call_calculator
)

graph_builder.add_node(
    "call_direct_llm",
    call_direct_llm
)

graph_builder.add_node(
    "call_web_search",
    call_web_search
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

graph_builder.set_finish_point(
    "call_calculator"
)

graph_builder.set_finish_point(
    "call_direct_llm"
)

graph_builder.set_finish_point(
    "call_web_search"
)

graph = graph_builder.compile()

result = graph.invoke(state)
print(result["answer"])
