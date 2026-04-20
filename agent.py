import os
import re
from typing import List, TypedDict

import chromadb
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from sentence_transformers import SentenceTransformer

load_dotenv()

# Shared knowledge base for the course assistant.
DOCUMENTS = [
    {
        "id": "doc_001",
        "topic": "LLM API Basics",
        "text": "LLM APIs accept prompts and return generated responses. Reliable assistants centralize model setup, use low temperature for consistency, and handle key/timeouts safely.",
    },
    {
        "id": "doc_002",
        "topic": "Function and Tool Calling",
        "text": "Tool calling is for deterministic actions such as prerequisite lookup and quiz generation. Routing should explicitly choose tool mode when the intent is structured.",
    },
    {
        "id": "doc_003",
        "topic": "Conversation Memory with Thread IDs",
        "text": "MemorySaver with thread_id keeps multi-turn continuity. Reusing the same thread_id preserves context, while a new thread_id starts a fresh conversation.",
    },
    {
        "id": "doc_004",
        "topic": "Embeddings and Semantic Retrieval",
        "text": "Embeddings map text into vectors for semantic similarity search. all-MiniLM-L6-v2 is a lightweight option for local retrieval workflows.",
    },
    {
        "id": "doc_005",
        "topic": "RAG Fundamentals",
        "text": "RAG combines retrieval and generation so answers are grounded in retrieved context. If context is missing, the assistant should say it clearly.",
    },
    {
        "id": "doc_006",
        "topic": "LangChain Components",
        "text": "LangChain helps structure prompts, model calls, and retrievers. It is often paired with LangGraph for orchestration.",
    },
    {
        "id": "doc_007",
        "topic": "LangGraph State and Node Design",
        "text": "LangGraph uses explicit state transitions across nodes. Clear node boundaries improve testability, debugging, and reliability.",
    },
    {
        "id": "doc_008",
        "topic": "Multi-Agent Patterns",
        "text": "Multi-agent setups separate responsibilities, but simpler route specialization is often enough for educational assistants.",
    },
    {
        "id": "doc_009",
        "topic": "Evaluation with Faithfulness",
        "text": "Faithfulness checks whether an answer is supported by context. Evaluation should be tracked across fixed test sets.",
    },
    {
        "id": "doc_010",
        "topic": "Prompt Injection and Safety Boundaries",
        "text": "Injection attempts must not override system constraints. The assistant should avoid fabricated claims and refuse unsafe behavior.",
    },
    {
        "id": "doc_011",
        "topic": "Streamlit Deployment Basics",
        "text": "Streamlit can expose the agent via chat UI. Session state stores thread_id and message history for continuity.",
    },
    {
        "id": "doc_012",
        "topic": "Capstone Testing Strategy",
        "text": "Capstone testing should include functional tests, memory continuity, and red-team checks with explicit pass-fail criteria.",
    },
]


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


FAITHFULNESS_THRESHOLD = 0.7
MAX_EVAL_RETRIES = 2


class CourseAgent:
    def __init__(self):
        groq_key = os.getenv("GROQ_API_KEY", "")
        if len(groq_key) < 10:
            raise RuntimeError("GROQ_API_KEY not found. Set it in .env.")

        self.llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")
        self.collection = self._build_collection()
        self.app = self._build_graph()

    def _build_collection(self):
        client = chromadb.Client()
        try:
            client.delete_collection("capstone_kb_agent")
        except Exception:
            pass
        collection = client.create_collection("capstone_kb_agent")

        texts = [d["text"] for d in DOCUMENTS]
        ids = [d["id"] for d in DOCUMENTS]
        embeddings = self.embedder.encode(texts).tolist()

        collection.add(
            documents=texts,
            embeddings=embeddings,
            ids=ids,
            metadatas=[{"topic": d["topic"]} for d in DOCUMENTS],
        )
        return collection

    def memory_node(self, state: CapstoneState) -> dict:
        msgs = state.get("messages", [])
        msgs = msgs + [{"role": "user", "content": state["question"]}]
        if len(msgs) > 6:
            msgs = msgs[-6:]
        return {"messages": msgs}

    def router_node(self, state: CapstoneState) -> dict:
        question = state["question"]
        messages = state.get("messages", [])
        recent = "; ".join(f"{m['role']}: {m['content'][:60]}" for m in messages[-3:-1]) or "none"

        prompt = f"""You are a strict router for an Agentic AI Course Assistant.

Choose exactly one option:
- retrieve: use knowledge base for concept/explanation questions
- memory_only: use recent conversation memory for follow-ups
- tool: use helper tool for prerequisites, revision, or quick quiz

Recent conversation: {recent}
Current question: {question}

Reply with ONLY one word: retrieve / memory_only / tool"""

        decision = self.llm.invoke(prompt).content.strip().lower()
        if "memory" in decision:
            decision = "memory_only"
        elif "tool" in decision or "quiz" in decision or "prereq" in decision:
            decision = "tool"
        else:
            decision = "retrieve"
        return {"route": decision}

    def retrieval_node(self, state: CapstoneState) -> dict:
        q_emb = self.embedder.encode([state["question"]]).tolist()
        results = self.collection.query(query_embeddings=q_emb, n_results=3)
        chunks = results["documents"][0]
        topics = [m["topic"] for m in results["metadatas"][0]]
        context = "\n\n---\n\n".join(f"[{topics[i]}]\n{chunks[i]}" for i in range(len(chunks)))
        return {"retrieved": context, "sources": topics}

    def skip_retrieval_node(self, state: CapstoneState) -> dict:
        return {"retrieved": "", "sources": []}

    def tool_node(self, state: CapstoneState) -> dict:
        question = state["question"].strip()
        lower_q = question.lower()

        prerequisite_map = {
            "day01": [],
            "day02": ["day01"],
            "day03": ["day01", "day02"],
            "day04": ["day01", "day03"],
            "day05": ["day02", "day04"],
            "day06": ["day03", "day05"],
            "day07": ["day06"],
            "day08": ["day05", "day06"],
            "day09": ["day08"],
            "day10": ["day04", "day09"],
            "day11": ["day10"],
            "day12": ["day11"],
            "day13": ["day08", "day10", "day11", "day12"],
        }

        quiz_bank = {
            "langgraph": [
                "What problem does StateGraph solve compared to plain chains?",
                "When should a node write to state vs compute locally?",
                "Why is conditional routing important for production agents?",
            ],
            "rag": [
                "Define RAG in one sentence.",
                "What causes retrieval mismatch in vector search?",
                "How does context grounding reduce hallucinations?",
            ],
            "memory": [
                "What is the purpose of thread_id?",
                "How is MemorySaver different from prompt-only memory?",
                "Why use a sliding window in conversation history?",
            ],
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

    def answer_node(self, state: CapstoneState) -> dict:
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
            if msg["role"] == "user":
                lc_msgs.append(HumanMessage(content=msg["content"]))
            else:
                lc_msgs.append(AIMessage(content=msg["content"]))
        lc_msgs.append(HumanMessage(content=question))

        response = self.llm.invoke(lc_msgs)
        return {"answer": response.content}

    def eval_node(self, state: CapstoneState) -> dict:
        answer = state.get("answer", "")
        context = state.get("retrieved", "")[:500]
        retries = state.get("eval_retries", 0)

        if not context:
            return {"faithfulness": 1.0, "eval_retries": retries + 1}

        prompt = f"""Rate faithfulness: does this answer use ONLY information from context?
Reply only with a number 0.0 to 1.0.
Context: {context}
Answer: {answer[:300]}"""

        raw = self.llm.invoke(prompt).content.strip()
        match = re.search(r"\b(0(?:\.\d+)?|1(?:\.0+)?)\b", raw)
        score = float(match.group(1)) if match else 0.5
        score = max(0.0, min(1.0, score))
        return {"faithfulness": score, "eval_retries": retries + 1}

    def save_node(self, state: CapstoneState) -> dict:
        messages = state.get("messages", [])
        messages = messages + [{"role": "assistant", "content": state["answer"]}]
        return {"messages": messages}

    @staticmethod
    def route_decision(state: CapstoneState) -> str:
        route = state.get("route", "retrieve")
        if route == "tool":
            return "tool"
        if route == "memory_only":
            return "skip"
        return "retrieve"

    @staticmethod
    def eval_decision(state: CapstoneState) -> str:
        score = state.get("faithfulness", 1.0)
        retries = state.get("eval_retries", 0)
        if score >= FAITHFULNESS_THRESHOLD or retries >= MAX_EVAL_RETRIES:
            return "save"
        return "answer"

    def _build_graph(self):
        graph = StateGraph(CapstoneState)
        graph.add_node("memory", self.memory_node)
        graph.add_node("router", self.router_node)
        graph.add_node("retrieve", self.retrieval_node)
        graph.add_node("skip", self.skip_retrieval_node)
        graph.add_node("tool", self.tool_node)
        graph.add_node("answer", self.answer_node)
        graph.add_node("eval", self.eval_node)
        graph.add_node("save", self.save_node)

        graph.set_entry_point("memory")
        graph.add_edge("memory", "router")
        graph.add_conditional_edges(
            "router",
            self.route_decision,
            {"retrieve": "retrieve", "skip": "skip", "tool": "tool"},
        )
        graph.add_edge("retrieve", "answer")
        graph.add_edge("skip", "answer")
        graph.add_edge("tool", "answer")
        graph.add_edge("answer", "eval")
        graph.add_conditional_edges(
            "eval",
            self.eval_decision,
            {"answer": "answer", "save": "save"},
        )
        graph.add_edge("save", END)

        return graph.compile(checkpointer=MemorySaver())

    def ask(self, question: str, thread_id: str = "default") -> dict:
        config = {"configurable": {"thread_id": thread_id}}
        return self.app.invoke({"question": question}, config=config)


def build_agent() -> CourseAgent:
    return CourseAgent()


if __name__ == "__main__":
    agent = build_agent()
    result = agent.ask("What is RAG?", thread_id="cli-demo")
    print(result.get("answer", "No answer"))
