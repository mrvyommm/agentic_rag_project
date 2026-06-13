# Agentic RAG project using Langgraph


from langgraph.graph import StateGraph
from ollama import embed, chat
import chromadb
from pathlib import Path
from dotenv import load_dotenv
import os
from tavily import TavilyClient
import json
from google import genai

BASE_DIR = Path(__file__).resolve().parent.parent.parent

load_dotenv()
tavily_client = TavilyClient(
    api_key=os.getenv("TAVILY_API_KEY")
)

gemini_client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
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

state = {
    "question": "",
    "plan": [],
    "tool": "",
    "context": [],
    "messages": [],
    "attempts": 0,
    "answer": ""
}


def ask_question(state):
    state["question"] = input(
        "Please ask your question  (or type exit) : ")

    return state


def question_router(state):
    if state["question"].lower() == "exit":
        return "exit"
    else:
        return "planner"


def planner(state):

    memory = get_memory(state)

    state["plan"] = create_plan(state)

    prompt = f"""

You are a routing agent.

Conversation History:
{memory}

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
    decision = call_local_llm(prompt)
    print("Decision: ", decision)

    try:
        parsed = json.loads(decision)
        state["tool"] = parsed["tool"]

    except Exception:
        print("Invalid JSON from planner")
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
        return "call_direct_llm"


def create_plan(state):
    prompt = f"""
You are a planning agent.
Question:
{state["question"]}

Break the task into steps 

Return JSON:

{{
        "plan":[
           "step1",
           "step2"
]

}}

Return raw JSON only.

Do NOT wrap in markdown.
Do NOT use ```json.
Do NOT explain.
Return only valid JSON.

"""
    plan = call_cloud_llm(prompt)
    parsed = json.loads(plan)
    state["plan"] = parsed["plan"]

    return state


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

    memory = get_memory(state)

    prompt = f"""

Conversation History:
{memory}

Question:
{state["question"]} 

You are a genious mentor:
-Answer the question in five bullet points.
-Keep it precise and clear.

"""

    state["answer"] = call_local_llm(prompt)

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

    state["answer"] = call_local_llm(prompt)

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

    decision = call_local_llm(prompt).strip().upper()

    print("Decision: ", decision)

    if "YES" in decision:
        state["enough_context"] = True

    else:
        state["enough_context"] = False
        state["question"] = decision

    return state


def context_router(state):
    if state["enough_context"]:
        return "final_retrieval"
    if state["attempts"] >= 3:
        return "final_retrieval"

    return "retrieve"


def final_retrieval(state):
    prompt = f"""
Question:
{state["question"]}

Context:
{state["context"]}

-Answer the question using context
-Answer in 5 bullet points
"""

    state["answer"] = call_local_llm(prompt)

    return state


def memory_update(state):
    state["messages"].append(
        {
            "role": "user",
            "content": state["question"]
        }
    )

    state["messages"].append(
        {
            "role": "assistant",
            "content": state["answer"]
        }
    )

    print("Answer: ", state["answer"])

    return state


def exit(state):

    # print("Messages: ", state["messages"])
    return state


def call_local_llm(prompt):
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


def call_cloud_llm(prompt):
    response = gemini_client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )
    answer = response.text
    print("Answer: ", answer)

    return answer


def get_memory(state):
    return state["messages"][-6:]


graph_builder = StateGraph(dict)


graph_builder.add_node(
    "ask_question",
    ask_question
)

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
    "final_retrieval",
    final_retrieval
)

graph_builder.add_node(
    "evaluate_context",
    evaluate_context
)

graph_builder.add_node(
    "memory_update",
    memory_update
)

graph_builder.add_node(
    "exit",
    exit
)


graph_builder.set_entry_point(
    "ask_question"
)

graph_builder.add_conditional_edges(
    "ask_question",
    question_router
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

graph_builder.add_edge(
    "final_retrieval",
    "memory_update"
)

graph_builder.add_edge(
    "call_calculator",
    "memory_update"
)

graph_builder.add_edge(
    "call_direct_llm",
    "memory_update"
)

graph_builder.add_edge(
    "call_web_search",
    "memory_update"
)

graph_builder.add_edge(
    "memory_update",
    "ask_question"
)

graph_builder.set_finish_point(
    "exit"
)


graph = graph_builder.compile()
graph.invoke(state)
