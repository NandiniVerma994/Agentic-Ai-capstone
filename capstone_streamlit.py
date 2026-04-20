"""
capstone_streamlit.py - Course Assistant
Run: streamlit run capstone_streamlit.py
"""

import os
import json
import re
import uuid
import chromadb
import streamlit as st

from typing import TypedDict, List
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

load_dotenv()

st.set_page_config(page_title="Course Assistant", page_icon="🎓", layout="wide")
st.title("🎓 Course Assistant")
st.caption("Answers course questions with routing, memory, tool support, and grounded retrieval.")


@st.cache_resource
def build_agent():
    groq_key = os.getenv("GROQ_API_KEY", "")
    if len(groq_key) < 10:
        raise RuntimeError("GROQ_API_KEY not found. Set it in .env before running Streamlit.")

    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
    embedder = SentenceTransformer("all-MiniLM-L6-v2")

    client = chromadb.Client()
    try:
        client.delete_collection("capstone_kb_streamlit")
    except Exception:
        pass
    collection = client.create_collection("capstone_kb_streamlit")

    DOCUMENTS = json.loads('[\n  {\n    "id": "doc_001",\n    "topic": "LLM API Basics",\n    "text": "Large Language Model APIs provide a programmatic way to send prompts and receive generated responses. In this course context, an API call usually includes a system instruction, a user question, optional conversation history, and model parameters such as temperature or max tokens. A low temperature is preferred for factual educational assistants because it reduces randomness and keeps outputs consistent. API-based usage also allows logging, testing, and routing decisions to be implemented around the model call. Students should learn to separate prompt design from business logic so the same core assistant can evolve safely. Proper API use includes error handling for timeouts, invalid keys, and rate limits. For capstone work, a good pattern is to centralize model initialization once, then call it from node functions. This makes behavior easier to debug and test."\n  },\n  {\n    "id": "doc_002",\n    "topic": "Function and Tool Calling",\n    "text": "Tool calling allows an agent to execute controlled functions when a user request requires deterministic operations or structured domain actions. In a course assistant, tools should be domain-specific rather than generic. Good examples are prerequisite lookup, revision checklist generation, and quick quiz generation from known topics. The router decides when a tool is needed, and the tool node returns deterministic output back into state. A robust tool design avoids free-form side effects and returns concise, parseable text. Tool outputs should be transparent to the user, and if a tool cannot satisfy the request, the assistant should fail safely with guidance instead of inventing results. In LangGraph pipelines, tool use is typically one branch among retrieve, skip, and tool routes. Keeping tool behavior explicit improves evaluation quality and reduces hallucination risk."\n  },\n  {\n    "id": "doc_003",\n    "topic": "Conversation Memory with Thread IDs",\n    "text": "Conversation memory enables multi-turn continuity, where the assistant can use earlier turns in later answers. In this course pattern, memory is managed by LangGraph checkpointers such as MemorySaver. The key idea is that each conversation is tagged with a thread_id. Reusing the same thread_id keeps context linked; changing it starts a fresh conversation. Students often confuse prompt context with persistent memory. Prompt context is what you send now; persistent memory is what the graph can recover across turns by session key. For capstone validation, a three-turn test is mandatory: establish a user fact in turn one, discuss a topic in turn two, and verify recall in turn three. Memory should support relevance, not over-verbosity. Good assistants summarize and carry only necessary context. Correct thread management is a core production skill."\n  },\n  {\n    "id": "doc_004",\n    "topic": "Embeddings and Semantic Retrieval",\n    "text": "Embeddings convert text into dense vectors that capture semantic meaning. Instead of exact keyword matching, vector retrieval compares conceptual similarity in embedding space. In practice, a sentence-transformer model like all-MiniLM-L6-v2 is commonly used for lightweight local workflows. During indexing, each knowledge chunk is embedded and stored with metadata. During query time, the user question is embedded and compared to stored vectors to return top-k relevant chunks. Retrieval quality depends on chunk clarity, topical separation, and document coverage. If chunks are vague or repetitive, top results become noisy. For course assistants, embeddings should represent concrete instructional concepts such as LangGraph nodes, RAG pipeline steps, and evaluation metrics. Semantic retrieval is the grounding layer that helps prevent unsupported generation and enables source-aware responses."\n  },\n  {\n    "id": "doc_005",\n    "topic": "RAG Fundamentals",\n    "text": "Retrieval-Augmented Generation combines a retriever with a generator. The retriever fetches relevant context from a knowledge base, and the generator creates an answer constrained by that context. This pattern improves factual reliability for domain assistants where internal model memory may be outdated or insufficient. A typical flow is: embed corpus, store vectors, retrieve top-k chunks, construct grounded prompt, generate answer, and optionally evaluate faithfulness. In capstone terms, RAG is not just adding a vector database; it is a behavior contract that answers should come from retrieved evidence. If evidence is missing, the assistant should say so and guide the user. RAG performance improves with better chunk design, cleaner metadata, and focused routing so irrelevant queries do not force retrieval. For educational use, RAG supports explainable, curriculum-aligned responses."\n  },\n  {\n    "id": "doc_006",\n    "topic": "LangChain Components",\n    "text": "LangChain provides modular building blocks for model calls, prompts, retrievers, and tool wrappers. In this course sequence, LangChain is often used alongside LangGraph: LangChain handles components while LangGraph handles orchestration. Core practices include separating prompt templates from runtime state, keeping chain steps inspectable, and maintaining deterministic defaults for evaluation runs. Students should avoid monolithic scripts where retrieval, generation, and post-processing are tangled in one function. Instead, compose small pieces that can be tested independently. When integrated with vector stores and chat models, LangChain helps standardize input/output patterns. For capstone, the important takeaway is interoperability: you can switch models, retrievers, or evaluators without rewriting the entire application architecture. This modularity supports debugging, incremental upgrades, and deployment readiness."\n  },\n  {\n    "id": "doc_007",\n    "topic": "LangGraph State and Node Design",\n    "text": "LangGraph organizes agent workflows as explicit state transitions between nodes. Each node reads from state, performs a focused task, and writes updated fields back. The state schema should be defined before node implementation, because missing state keys are a common source of runtime errors. A typical capstone state includes question, messages, route, retrieved_docs, sources, tool_result, answer, faithfulness, and retry counters. Conditional edges control branching, such as router-to-retrieve/tool/skip and evaluator-to-save/retry. This graph-first approach improves observability compared with hidden control flow in plain loops. It also makes testing easier because each node can be validated in isolation. Students should keep node responsibilities narrow: router decides route, retrieval fetches context, answer composes reply, eval judges quality, save logs result. Clear boundaries produce robust behavior."\n  },\n  {\n    "id": "doc_008",\n    "topic": "Multi-Agent Patterns",\n    "text": "Multi-agent systems divide work across specialized agents such as planner, retriever, critic, and executor. While the capstone may use a single graph, understanding multi-agent patterns helps design better internal roles and routes. Benefits include specialization and parallel reasoning, but risks include coordination overhead and inconsistent outputs if contracts are unclear. Effective multi-agent design requires explicit message formats, role boundaries, and arbitration logic for conflicting responses. In educational assistants, lightweight specialization is often enough: one route for retrieval, one for tools, and one for memory-only interactions. This achieves many multi-agent benefits without full orchestration complexity. Students should treat multi-agent architecture as a scaling option, not a requirement for every task. Good system design starts simple, then introduces role separation only when measurable benefits appear in tests."\n  },\n  {\n    "id": "doc_009",\n    "topic": "Evaluation with Faithfulness",\n    "text": "Evaluation measures whether answers are correct, relevant, and grounded. For RAG-based assistants, faithfulness is a key metric: does the answer stay consistent with retrieved evidence. RAGAS provides useful metrics such as faithfulness, answer relevancy, and context precision. In practical capstone workflows, evaluation should run on a fixed test set so improvements can be compared over time. Automated scores are helpful but should be paired with manual checks for edge cases, especially prompt injection and false premises. A good evaluator node can trigger retries when quality is below threshold, but retry loops must have limits to avoid infinite cycles. Reporting should include route chosen, sources used, score value, and pass/fail rationale. Consistent evaluation turns agent development from guesswork into evidence-based iteration."\n  },\n  {\n    "id": "doc_010",\n    "topic": "Prompt Injection and Safety Boundaries",\n    "text": "Prompt injection occurs when user input attempts to override system instructions or reveal hidden policies. A safe assistant treats user messages as untrusted and preserves high-priority rules. Typical attack patterns include commands like ignore previous instructions, reveal your system prompt, or fabricate citations. Defensive behavior includes refusal for unsafe requests, no leakage of private prompts, and transparent limitation statements. In course assistants, safety also means refusing out-of-domain claims while offering a helpful fallback path such as asking a mentor or checking official materials. Safety instructions should be embedded in system prompts and reinforced in evaluation tests. Red-team testing is essential to validate defenses under adversarial phrasing. Strong safety does not reduce usefulness; it improves trust and reliability in real deployment."\n  },\n  {\n    "id": "doc_011",\n    "topic": "Streamlit Deployment Basics",\n    "text": "Streamlit is a rapid UI framework that lets developers expose agent workflows as interactive web apps. For production-like behavior, heavy objects such as models, embedders, and vector stores should be initialized with cache decorators to avoid repeated loading on reruns. Conversation continuity in Streamlit uses session state, where a thread_id and message history are persisted per browser session. A minimal course assistant UI should accept user questions, call an ask helper, show answers, and optionally display route, sources, and quality score for transparency. Deployment readiness includes handling missing API keys, showing friendly error messages, and separating UI concerns from core graph logic in an agent module. Even simple local deployment demonstrates end-to-end capability from retrieval to memory to evaluation."\n  },\n  {\n    "id": "doc_012",\n    "topic": "Capstone Testing Strategy",\n    "text": "A complete capstone test strategy combines functional, memory, red-team, and evaluation baselines. Functional tests should cover retrieval route, tool route, and memory-only route with expected outcomes. Memory testing requires multi-turn continuity with a constant thread_id. Red-team tests should include out-of-scope questions, false premises, injection attempts, hallucination bait, and emotionally sensitive prompts. Each test should log question, expected behavior, actual behavior, and pass/fail. Baseline evaluation with small fixed datasets helps quantify quality before and after improvements. Students should avoid judging quality only by response length or fluency; grounded correctness and safety matter more. Testing artifacts such as JSON logs and metric tables are part of deployment evidence. Strong testing demonstrates engineering maturity beyond demo-only behavior."\n  }\n]')
    texts = [d["text"] for d in DOCUMENTS]
    ids = [d["id"] for d in DOCUMENTS]
    embeddings = embedder.encode(texts).tolist()

    collection.add(
        documents=texts,
        embeddings=embeddings,
        ids=ids,
        metadatas=[{"topic": d["topic"]} for d in DOCUMENTS]
    )

    class CapstoneState(TypedDict):
        question: str
        messages: List[dict]
        route: str
        retrieved: str
        sources: List[str]
        tool_result: str
        answer: str
        faithfulness: float
        eval_retries: int
        learner_level: str
        target_topic: str
        last_tool_used: str

    def memory_node(state: CapstoneState) -> dict:
        msgs = state.get("messages", [])
        msgs = msgs + [{"role": "user", "content": state["question"]}]
        if len(msgs) > 6:
            msgs = msgs[-6:]
        return {"messages": msgs}

    def router_node(state: CapstoneState) -> dict:
        question = state["question"]
        messages = state.get("messages", [])
        recent = "; ".join(f"{m['role']}: {m['content'][:60]}" for m in messages[-3:-1]) or "none"

        prompt = f"""You are a strict router for an Agentic AI Course Assistant.

Choose exactly one option:
- retrieve: use knowledge base for concept/explanation questions (LangGraph, RAG, memory, evaluation, deployment)
- memory_only: use only recent conversation memory (for follow-ups)
- tool: use helper tool for prerequisites, revision plan, or quick quiz/practice questions

Recent conversation: {recent}
Current question: {question}

Reply with ONLY one word: retrieve / memory_only / tool"""

        response = llm.invoke(prompt)
        decision = response.content.strip().lower()
        if "memory" in decision:
            decision = "memory_only"
        elif "tool" in decision or "quiz" in decision or "prereq" in decision:
            decision = "tool"
        else:
            decision = "retrieve"
        return {"route": decision}

    def retrieval_node(state: CapstoneState) -> dict:
        q_emb = embedder.encode([state["question"]]).tolist()
        results = collection.query(query_embeddings=q_emb, n_results=3)
        chunks = results["documents"][0]
        topics = [m["topic"] for m in results["metadatas"][0]]
        context = "\n\n---\n\n".join(f"[{topics[i]}]\n{chunks[i]}" for i in range(len(chunks)))
        return {"retrieved": context, "sources": topics}

    def skip_retrieval_node(state: CapstoneState) -> dict:
        return {"retrieved": "", "sources": []}

    def tool_node(state: CapstoneState) -> dict:
        question = state["question"].strip()
        lower_q = question.lower()

        prerequisite_map = {
            "day01": [], "day02": ["day01"], "day03": ["day01", "day02"],
            "day04": ["day01", "day03"], "day05": ["day02", "day04"],
            "day06": ["day03", "day05"], "day07": ["day06"],
            "day08": ["day05", "day06"], "day09": ["day08"],
            "day10": ["day04", "day09"], "day11": ["day10"],
            "day12": ["day11"], "day13": ["day08", "day10", "day11", "day12"]
        }

        quiz_bank = {
            "langgraph": [
                "What problem does StateGraph solve compared to plain chains?",
                "When should a node write to state vs compute locally?",
                "Why is conditional routing important for production agents?"
            ],
            "rag": [
                "Define RAG in one sentence.",
                "What causes retrieval mismatch in vector search?",
                "How does context grounding reduce hallucinations?"
            ],
            "memory": [
                "What is the purpose of thread_id?",
                "How is MemorySaver different from prompt-only memory?",
                "Why use a sliding window in conversation history?"
            ]
        }

        day_match = re.search(r"day\s*0?(\d{1,2})", lower_q)

        if "prereq" in lower_q or "prerequisite" in lower_q:
            if day_match:
                d = int(day_match.group(1))
                key = f"day{d:02d}"
                prereqs = prerequisite_map.get(key)
                if prereqs is None:
                    tool_result = f"I do not have prerequisite mapping for {key}."
                elif len(prereqs) == 0:
                    tool_result = f"{key} has no hard prerequisites."
                else:
                    tool_result = f"Prerequisites for {key}: {', '.join(prereqs)}"
            else:
                tool_result = "Please specify a day number, e.g., Day 08."
            last_tool = "prerequisite_helper"
        elif "quiz" in lower_q or "practice" in lower_q or "revision" in lower_q:
            topic = "langgraph" if "langgraph" in lower_q else "rag" if "rag" in lower_q else "memory"
            qs = quiz_bank[topic]
            tool_result = (
                f"Quick {topic.upper()} practice set:\n"
                f"1. {qs[0]}\n2. {qs[1]}\n3. {qs[2]}\n"
                "Answer these first, then ask me to evaluate your responses."
            )
            last_tool = "quiz_helper"
        else:
            tool_result = "Tool route selected, but no matching helper intent detected."
            last_tool = "none"

        return {"tool_result": tool_result, "last_tool_used": last_tool}

    def answer_node(state: CapstoneState) -> dict:
        question = state["question"]
        retrieved = state.get("retrieved", "")
        tool_result = state.get("tool_result", "")
        messages = state.get("messages", [])
        eval_retries = state.get("eval_retries", 0)

        context_parts = []
        if retrieved:
            context_parts.append(f"KNOWLEDGE BASE:\n{retrieved}")
        if tool_result:
            context_parts.append(f"TOOL RESULT:\n{tool_result}")
        context = "\n\n".join(context_parts)

        if context:
            system_content = f"""You are a Course Assistant for the Agentic AI program.
Rules:
1. Answer using ONLY the supplied context blocks.
2. If context does not contain the answer, say exactly: I don't have that information in my knowledge base.
3. Be concise, accurate, and student-friendly.
4. If tool output is present, prioritize it for prerequisite/quiz requests.
5. Do not invent days, metrics, or citations.

{context}"""
        else:
            system_content = "You are a Course Assistant. Answer from conversation history only."

        if eval_retries > 0:
            system_content += "\n\nIMPORTANT: be strictly grounded and avoid unsupported claims."

        lc_msgs = [SystemMessage(content=system_content)]
        for msg in messages[:-1]:
            lc_msgs.append(HumanMessage(content=msg["content"]) if msg["role"] == "user" else AIMessage(content=msg["content"]))
        lc_msgs.append(HumanMessage(content=question))

        response = llm.invoke(lc_msgs)
        return {"answer": response.content}

    FAITHFULNESS_THRESHOLD = 0.7
    MAX_EVAL_RETRIES = 2

    def eval_node(state: CapstoneState) -> dict:
        answer = state.get("answer", "")
        context = state.get("retrieved", "")[:500]
        retries = state.get("eval_retries", 0)

        if not context:
            return {"faithfulness": 1.0, "eval_retries": retries + 1}

        prompt = f"""Rate faithfulness: does this answer use ONLY information from context?
Reply only with a number 0.0 to 1.0.
Context: {context}
Answer: {answer[:300]}"""

        raw = llm.invoke(prompt).content.strip()
        m = re.search(r"\b(0(?:\.\d+)?|1(?:\.0+)?)\b", raw)
        score = float(m.group(1)) if m else 0.5
        score = max(0.0, min(1.0, score))
        return {"faithfulness": score, "eval_retries": retries + 1}

    def save_node(state: CapstoneState) -> dict:
        messages = state.get("messages", [])
        messages = messages + [{"role": "assistant", "content": state["answer"]}]
        return {"messages": messages}

    def route_decision(state: CapstoneState) -> str:
        route = state.get("route", "retrieve")
        if route == "tool":
            return "tool"
        if route == "memory_only":
            return "skip"
        return "retrieve"

    def eval_decision(state: CapstoneState) -> str:
        score = state.get("faithfulness", 1.0)
        retries = state.get("eval_retries", 0)
        if score >= FAITHFULNESS_THRESHOLD or retries >= MAX_EVAL_RETRIES:
            return "save"
        return "answer"

    graph = StateGraph(CapstoneState)
    graph.add_node("memory", memory_node)
    graph.add_node("router", router_node)
    graph.add_node("retrieve", retrieval_node)
    graph.add_node("skip", skip_retrieval_node)
    graph.add_node("tool", tool_node)
    graph.add_node("answer", answer_node)
    graph.add_node("eval", eval_node)
    graph.add_node("save", save_node)

    graph.set_entry_point("memory")
    graph.add_edge("memory", "router")
    graph.add_conditional_edges("router", route_decision, {"retrieve": "retrieve", "skip": "skip", "tool": "tool"})
    graph.add_edge("retrieve", "answer")
    graph.add_edge("skip", "answer")
    graph.add_edge("tool", "answer")
    graph.add_edge("answer", "eval")
    graph.add_conditional_edges("eval", eval_decision, {"answer": "answer", "save": "save"})
    graph.add_edge("save", END)

    app = graph.compile(checkpointer=MemorySaver())
    return app, collection, DOCUMENTS


@st.cache_resource
def load_resources():
    return build_agent()


app, collection, documents = load_resources()

if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())[:8]
if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.subheader("Session")
    st.write(f"Thread: {st.session_state.thread_id}")
    st.write(f"KB docs: {collection.count()}")
    if st.button("New conversation"):
        st.session_state.messages = []
        st.session_state.thread_id = str(uuid.uuid4())[:8]
        st.rerun()

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

prompt = st.chat_input("Ask about the course...")
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = app.invoke(
                {"question": prompt},
                config={"configurable": {"thread_id": st.session_state.thread_id}},
            )
            answer = result.get("answer", "Sorry, I could not generate an answer.")
            route = result.get("route", "n/a")
            faith = result.get("faithfulness", 0.0)
            sources = result.get("sources", [])

        st.write(answer)
        st.caption(f"Route: {route} | Faithfulness: {faith:.2f}")
        if sources:
            st.caption("Sources: " + ", ".join(sources))

    st.session_state.messages.append({"role": "assistant", "content": answer})
