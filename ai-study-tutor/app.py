"""
AI Study Mentor (LangGraph + Streamlit)
=======================================

Features:
- Tracks learner knowledge state by topic
- Asks adaptive quiz questions based on mastery
- Explains weak topics
- Persists state across turns in Streamlit session
"""

from __future__ import annotations

import json
import os
import random
import re
from typing import Dict, List, Literal, TypedDict

import streamlit as st
from dotenv import load_dotenv
from langgraph.graph import END, START, StateGraph
from langchain_openai import ChatOpenAI


TOPICS = [
    "python",
    "statistics",
    "machine learning",
    "linear algebra",
    "data structures",
]


QUESTION_BANK: Dict[str, Dict[str, List[Dict[str, str | List[str]]]]] = {
    "python": {
        "easy": [
            {
                "question": "Which data type is immutable in Python?",
                "options": ["list", "set", "dict", "tuple"],
                "answer": "tuple",
                "explanation": "Tuples cannot be changed after creation.",
            }
        ],
        "medium": [
            {
                "question": "What is the output of `len({1, 1, 2, 3})`?",
                "options": ["4", "3", "2", "Error"],
                "answer": "3",
                "explanation": "Sets remove duplicates, so the set is {1,2,3}.",
            }
        ],
        "hard": [
            {
                "question": "What does a Python generator primarily optimize?",
                "options": [
                    "Disk usage",
                    "Memory usage",
                    "CPU clock speed",
                    "Network latency",
                ],
                "answer": "Memory usage",
                "explanation": "Generators yield items lazily without storing all values.",
            }
        ],
    },
    "statistics": {
        "easy": [
            {
                "question": "Which measure is most affected by outliers?",
                "options": ["Median", "Mode", "Mean", "IQR"],
                "answer": "Mean",
                "explanation": "The mean shifts strongly when extreme values are present.",
            }
        ],
        "medium": [
            {
                "question": "If p-value < 0.05, what is a common decision?",
                "options": [
                    "Reject null hypothesis",
                    "Accept null hypothesis as true",
                    "Increase sample variance",
                    "Stop collecting data immediately",
                ],
                "answer": "Reject null hypothesis",
                "explanation": "It suggests data is unlikely under the null hypothesis.",
            }
        ],
        "hard": [
            {
                "question": "Which distribution often models sample means?",
                "options": [
                    "Uniform distribution",
                    "Poisson distribution",
                    "Normal distribution",
                    "Cauchy distribution",
                ],
                "answer": "Normal distribution",
                "explanation": "By the central limit theorem, sample means tend toward normality.",
            }
        ],
    },
    "machine learning": {
        "easy": [
            {
                "question": "Which task predicts a continuous value?",
                "options": ["Classification", "Regression", "Clustering", "Ranking"],
                "answer": "Regression",
                "explanation": "Regression outputs numeric continuous predictions.",
            }
        ],
        "medium": [
            {
                "question": "What is overfitting?",
                "options": [
                    "Model performs well on train but poorly on test data",
                    "Model has low train and low test accuracy",
                    "Model uses too little data",
                    "Model always underestimates",
                ],
                "answer": "Model performs well on train but poorly on test data",
                "explanation": "Overfitting captures noise from training data.",
            }
        ],
        "hard": [
            {
                "question": "Which method directly helps reduce variance?",
                "options": [
                    "Bagging",
                    "Increasing learning rate",
                    "Removing validation set",
                    "Reducing data",
                ],
                "answer": "Bagging",
                "explanation": "Bagging averages multiple models to reduce variance.",
            }
        ],
    },
    "linear algebra": {
        "easy": [
            {
                "question": "A matrix with same rows and columns is called?",
                "options": [
                    "Singular matrix",
                    "Orthogonal matrix",
                    "Square matrix",
                    "Diagonal matrix",
                ],
                "answer": "Square matrix",
                "explanation": "Square matrices have dimension n x n.",
            }
        ],
        "medium": [
            {
                "question": "What indicates that vectors are orthogonal?",
                "options": [
                    "Dot product is 1",
                    "Dot product is 0",
                    "Magnitudes are equal",
                    "Cross product is 0",
                ],
                "answer": "Dot product is 0",
                "explanation": "Orthogonal vectors have zero dot product.",
            }
        ],
        "hard": [
            {
                "question": "If det(A)=0, A is:",
                "options": ["Invertible", "Singular", "Orthogonal", "Positive definite"],
                "answer": "Singular",
                "explanation": "A zero determinant means no inverse exists.",
            }
        ],
    },
    "data structures": {
        "easy": [
            {
                "question": "Which data structure follows FIFO?",
                "options": ["Stack", "Queue", "Tree", "Heap"],
                "answer": "Queue",
                "explanation": "First In, First Out behavior defines queues.",
            }
        ],
        "medium": [
            {
                "question": "Average lookup time in a hash table is typically:",
                "options": ["O(1)", "O(log n)", "O(n)", "O(n log n)"],
                "answer": "O(1)",
                "explanation": "With good hashing and low collisions, lookup is near constant.",
            }
        ],
        "hard": [
            {
                "question": "Which traversal uses a queue in trees?",
                "options": ["Inorder", "Preorder", "Postorder", "Level-order"],
                "answer": "Level-order",
                "explanation": "Breadth-first (level-order) traversal uses a queue.",
            }
        ],
    },
}


class MentorState(TypedDict):
    user_input: str
    intent: Literal["learn", "quiz", "status", "awaiting_answer"]
    active_topic: str
    weak_topic: str
    knowledge: Dict[str, float]
    assistant_response: str
    awaiting_answer: bool
    current_question: str
    current_options: List[str]
    current_answer: str
    current_explanation: str
    was_correct: bool
    quiz_attempts: int
    quiz_correct: int


def fallback_lesson(topic: str, mastery: float, weak: str) -> str:
    return (
        f"Topic focus: **{topic.title()}**\n\n"
        f"Current mastery estimate: **{mastery:.0%}**.\n"
        f"Your weakest area overall is **{weak.title()}**.\n\n"
        f"Quick explanation:\n"
        f"- Core idea: understand the fundamentals of {topic} before memorizing formulas.\n"
        f"- Practice loop: learn concept -> solve 3 short problems -> explain back in your own words.\n"
        f"- Next step: ask me for a quiz on {topic} by typing `quiz {topic}`."
    )


def fallback_quiz(topic: str, difficulty: str) -> Dict[str, str | List[str]]:
    item = random.choice(QUESTION_BANK[topic][difficulty])
    return {
        "question": str(item["question"]),
        "options": [str(opt) for opt in item["options"]],
        "answer": str(item["answer"]),
        "explanation": str(item["explanation"]),
    }


@st.cache_resource
def get_llm():
    if not os.getenv("OPENAI_API_KEY"):
        return None
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    return ChatOpenAI(model=model, temperature=0.2)


def llm_lesson(topic: str, mastery: float, weak: str) -> str | None:
    llm = get_llm()
    if llm is None:
        return None
    prompt = (
        "You are an academic AI study mentor.\n"
        f"Topic: {topic}\n"
        f"Mastery score: {mastery:.2f} (0-1)\n"
        f"Weakest overall topic: {weak}\n"
        "Write a concise coaching response in Markdown with:\n"
        "1) one-paragraph explanation,\n"
        "2) 3 bullet action plan,\n"
        "3) one short self-check question.\n"
        "Keep it under 140 words."
    )
    try:
        return llm.invoke(prompt).content.strip()
    except Exception:
        return None


def llm_quiz(topic: str, difficulty: str) -> Dict[str, str | List[str]] | None:
    llm = get_llm()
    if llm is None:
        return None
    prompt = (
        "Generate one multiple-choice quiz item as valid JSON only.\n"
        f"Topic: {topic}\n"
        f"Difficulty: {difficulty}\n"
        'Return keys exactly: "question", "options", "answer", "explanation".\n'
        'Rules: "options" must contain exactly 4 strings, "answer" must match one option exactly.'
    )
    try:
        raw = llm.invoke(prompt).content.strip()
        data = json.loads(raw)
        if not isinstance(data, dict):
            return None
        question = str(data.get("question", "")).strip()
        options = data.get("options", [])
        answer = str(data.get("answer", "")).strip()
        explanation = str(data.get("explanation", "")).strip()
        if not question or not explanation or not isinstance(options, list) or len(options) != 4:
            return None
        norm_options = [str(opt).strip() for opt in options]
        if answer not in norm_options:
            return None
        return {
            "question": question,
            "options": norm_options,
            "answer": answer,
            "explanation": explanation,
        }
    except Exception:
        return None


def normalize_topic(text: str) -> str:
    lower = text.lower()
    aliases = {
        "ml": "machine learning",
        "ai": "machine learning",
        "stats": "statistics",
        "py": "python",
        "dsa": "data structures",
        "algebra": "linear algebra",
    }
    for key, topic in aliases.items():
        if re.search(rf"\b{re.escape(key)}\b", lower):
            return topic
    for topic in TOPICS:
        if topic in lower:
            return topic
    return ""


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def choose_difficulty(mastery: float) -> str:
    if mastery < 0.4:
        return "easy"
    if mastery < 0.7:
        return "medium"
    return "hard"


def detect_intent(state: MentorState) -> MentorState:
    text = state["user_input"].strip().lower()
    if state["awaiting_answer"]:
        state["intent"] = "awaiting_answer"
        return state

    if any(word in text for word in ["quiz", "test", "mcq", "question"]):
        state["intent"] = "quiz"
    elif any(word in text for word in ["progress", "status", "report", "score"]):
        state["intent"] = "status"
    else:
        state["intent"] = "learn"
    return state


def knowledge_tracker(state: MentorState) -> MentorState:
    topic = normalize_topic(state["user_input"])
    if topic:
        state["active_topic"] = topic
    if not state["active_topic"]:
        state["active_topic"] = "python"

    state["weak_topic"] = min(state["knowledge"], key=state["knowledge"].get)
    return state


def lesson_generator(state: MentorState) -> MentorState:
    topic = state["active_topic"]
    weak = state["weak_topic"]
    mastery = state["knowledge"][topic]
    llm_response = llm_lesson(topic, mastery, weak)
    state["assistant_response"] = llm_response or fallback_lesson(topic, mastery, weak)
    return state


def quiz_node(state: MentorState) -> MentorState:
    topic = state["active_topic"]
    mastery = state["knowledge"][topic]
    difficulty = choose_difficulty(mastery)
    item = llm_quiz(topic, difficulty) or fallback_quiz(topic, difficulty)

    question = str(item["question"])
    options = [str(opt) for opt in item["options"]]
    answer = str(item["answer"])
    explanation = str(item["explanation"])

    option_lines = "\n".join([f"{idx + 1}. {opt}" for idx, opt in enumerate(options)])
    state["assistant_response"] = (
        f"Adaptive Quiz: **{topic.title()}** ({difficulty})\n\n"
        f"{question}\n\n{option_lines}\n\n"
        "Reply with the option number or exact option text."
    )
    state["current_question"] = question
    state["current_options"] = options
    state["current_answer"] = answer
    state["current_explanation"] = explanation
    state["awaiting_answer"] = True
    return state


def evaluation_node(state: MentorState) -> MentorState:
    user_text = state["user_input"].strip()
    options = state["current_options"]
    answer = state["current_answer"]
    explanation = state["current_explanation"]
    topic = state["active_topic"]

    selected = user_text
    if user_text.isdigit():
        idx = int(user_text) - 1
        if 0 <= idx < len(options):
            selected = options[idx]

    correct = selected.strip().lower() == answer.strip().lower()
    state["was_correct"] = correct
    state["quiz_attempts"] += 1
    if correct:
        state["quiz_correct"] += 1

    state["assistant_response"] = (
        f"{'Correct' if correct else 'Not quite'}.\n\n"
        f"Answer: **{answer}**\n"
        f"Why: {explanation}\n\n"
        f"Type `quiz {topic}` for another question or ask `status`."
    )
    state["awaiting_answer"] = False
    return state


def memory_update(state: MentorState) -> MentorState:
    if state["intent"] == "awaiting_answer":
        delta = 0.1 if state["was_correct"] else -0.08
        topic = state["active_topic"]
        state["knowledge"][topic] = clamp(state["knowledge"][topic] + delta)
        state["weak_topic"] = min(state["knowledge"], key=state["knowledge"].get)
        return state

    if state["intent"] == "status":
        sorted_topics = sorted(state["knowledge"].items(), key=lambda kv: kv[1])
        lines = [f"- {name.title()}: {score:.0%}" for name, score in sorted_topics]
        accuracy = (
            (state["quiz_correct"] / state["quiz_attempts"]) if state["quiz_attempts"] else 0.0
        )
        state["assistant_response"] = (
            "Progress Report\n\n"
            f"Quiz accuracy: **{accuracy:.0%}** ({state['quiz_correct']}/{state['quiz_attempts']})\n"
            f"Weakest topic: **{state['weak_topic'].title()}**\n\n"
            "Mastery by topic:\n" + "\n".join(lines)
        )
    return state


def route_after_intent(state: MentorState) -> str:
    if state["intent"] == "awaiting_answer":
        return "evaluation"
    if state["intent"] in ("quiz", "learn"):
        return "knowledge_tracker"
    return "memory_update"


def route_after_tracker(state: MentorState) -> str:
    if state["intent"] == "quiz":
        return "quiz"
    return "lesson_generator"


@st.cache_resource
def build_graph():
    graph = StateGraph(MentorState)
    graph.add_node("intent", detect_intent)
    graph.add_node("knowledge_tracker", knowledge_tracker)
    graph.add_node("lesson_generator", lesson_generator)
    graph.add_node("quiz", quiz_node)
    graph.add_node("evaluation", evaluation_node)
    graph.add_node("memory_update", memory_update)

    graph.add_edge(START, "intent")
    graph.add_conditional_edges(
        "intent",
        route_after_intent,
        {
            "knowledge_tracker": "knowledge_tracker",
            "evaluation": "evaluation",
            "memory_update": "memory_update",
        },
    )
    graph.add_conditional_edges(
        "knowledge_tracker",
        route_after_tracker,
        {"quiz": "quiz", "lesson_generator": "lesson_generator"},
    )
    graph.add_edge("lesson_generator", "memory_update")
    graph.add_edge("quiz", "memory_update")
    graph.add_edge("evaluation", "memory_update")
    graph.add_edge("memory_update", END)
    return graph.compile()


def initial_state() -> MentorState:
    return {
        "user_input": "",
        "intent": "learn",
        "active_topic": "python",
        "weak_topic": "python",
        "knowledge": {topic: 0.5 for topic in TOPICS},
        "assistant_response": (
            "I am your AI Study Mentor. Ask for a concept explanation, type `quiz python`, "
            "or ask `status` for your progress."
        ),
        "awaiting_answer": False,
        "current_question": "",
        "current_options": [],
        "current_answer": "",
        "current_explanation": "",
        "was_correct": False,
        "quiz_attempts": 0,
        "quiz_correct": 0,
    }


def initialize_session_state() -> None:
    if "mentor_state" not in st.session_state:
        st.session_state["mentor_state"] = initial_state()
    if "display_history" not in st.session_state:
        st.session_state["display_history"] = [
            {"role": "assistant", "content": st.session_state["mentor_state"]["assistant_response"]}
        ]


def main() -> None:
    st.set_page_config(page_title="AI Study Mentor", layout="wide")
    st.title("AI Study Mentor")
    st.caption("LangGraph demo: adaptive questions, weak-topic coaching, and persistent learning state.")

    load_dotenv()
    initialize_session_state()
    app_graph = build_graph()

    with st.sidebar:
        st.subheader("Knowledge Tracker")
        for topic, score in st.session_state["mentor_state"]["knowledge"].items():
            st.progress(score, text=f"{topic.title()} ({score:.0%})")
        st.markdown(
            f"**Weakest topic:** {st.session_state['mentor_state']['weak_topic'].title()}"
        )
        if st.button("Reset Mentor State"):
            st.session_state["mentor_state"] = initial_state()
            st.session_state["display_history"] = [
                {
                    "role": "assistant",
                    "content": st.session_state["mentor_state"]["assistant_response"],
                }
            ]
            st.rerun()
        st.divider()
        st.markdown(
            f"**LLM mode:** {'ON' if get_llm() is not None else 'OFF (set OPENAI_API_KEY)'}"
        )

    with st.expander("Study Mentor Workflow Graph", expanded=False):
        st.graphviz_chart(
            """
            digraph MentorFlow {
                rankdir=LR;
                node [shape=box, style="rounded,filled", fillcolor="#1f2430", fontcolor="white"];
                start [label="START", shape=circle, fillcolor="#2f9e44"];
                end [label="END", shape=doublecircle, fillcolor="#c92a2a"];
                intent [label="Intent Node"];
                knowledge_tracker [label="Knowledge Tracker Node"];
                lesson_generator [label="Lesson Generator Node"];
                quiz [label="Quiz Node"];
                evaluation [label="Evaluation Node"];
                memory_update [label="Memory Update Node"];
                start -> intent;
                intent -> knowledge_tracker [label="learn / quiz"];
                intent -> evaluation [label="awaiting answer"];
                intent -> memory_update [label="status"];
                knowledge_tracker -> lesson_generator [label="learn"];
                knowledge_tracker -> quiz [label="quiz"];
                lesson_generator -> memory_update;
                quiz -> memory_update;
                evaluation -> memory_update;
                memory_update -> end;
            }
            """
        )

    for msg in st.session_state["display_history"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_input = st.chat_input("Ask to learn, request a quiz, or check status.")
    if user_input:
        st.session_state["display_history"].append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        state = dict(st.session_state["mentor_state"])
        state["user_input"] = user_input
        updated_state = app_graph.invoke(state)
        st.session_state["mentor_state"] = updated_state

        response = updated_state["assistant_response"]
        st.session_state["display_history"].append({"role": "assistant", "content": response})
        with st.chat_message("assistant"):
            st.markdown(response)


if __name__ == "__main__":
    main()
