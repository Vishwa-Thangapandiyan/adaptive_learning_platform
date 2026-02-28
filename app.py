import hashlib
import random
import re
from collections import Counter

import pandas as pd
from flask import Flask, abort, render_template, request

from model import predict_difficulty, vectorizer

app = Flask(__name__)

DATA_PATH = "datasets/study_data.csv"


def normalize_dataset_difficulty(value):
    label = str(value).strip().lower()
    mapping = {"basic": "easy", "advanced": "hard"}
    return mapping.get(label, label if label in {"easy", "medium", "hard"} else "medium")


def normalize_model_difficulty(value):
    label = str(value).strip().lower()
    if label in {"basic", "easy"}:
        return "basic"
    if label in {"advanced", "hard", "medium"}:
        return "advanced"
    return "advanced"


def load_data():
    df = pd.read_csv(DATA_PATH).dropna(subset=["subject", "topic", "text", "difficulty"]).copy()
    df["subject"] = df["subject"].astype(str).str.strip()
    df["topic"] = df["topic"].astype(str).str.strip()
    df["text"] = df["text"].astype(str).str.strip()
    df["difficulty"] = df["difficulty"].apply(normalize_dataset_difficulty)
    df["preview"] = df["text"].str.slice(0, 150) + df["text"].apply(lambda t: "..." if len(t) > 150 else "")
    return df


DATA = load_data()
TOPIC_LOOKUP = {str(row["topic"]).lower(): row for _, row in DATA.iterrows()}


IGNORE_WORDS = {
    "concept",
    "terms",
    "topic",
    "theory",
    "reduce",
    "physical",
    "physics",
    "chemistry",
    "biology",
    "computer",
    "science",
    "begins",
    "begin",
    "quantities",
    "quantity",
    "calculation",
    "calculations",
}


def extract_keywords_from_text(text, top_n=5):
    text_vector = vectorizer.transform([str(text)])
    row = text_vector.toarray().flatten()
    feature_names = vectorizer.get_feature_names_out()
    top_indices = row.argsort()[::-1][:top_n]
    return [feature_names[i] for i in top_indices if row[i] > 0]


def clean_keyword(word):
    word = word.lower().strip()
    word = re.sub(r"[^a-z]", "", word)
    if len(word) < 5:
        return None
    if word in IGNORE_WORDS:
        return None
    if not re.search(r"[aeiou]", word):
        return None
    return word


def generate_study_tips(text, score=None):
    raw_keywords = extract_keywords_from_text(text)
    seen = set()
    final_keywords = []
    for phrase in raw_keywords:
        words = phrase.lower().split()
        for word in words:
            cleaned = clean_keyword(word)
            if cleaned and cleaned not in seen:
                seen.add(cleaned)
                final_keywords.append(cleaned)
    tips = [f"Revise the basic concept of {word}." for word in final_keywords]
    return tips[:5]


def generate_study_plan(difficulty):
    difficulty = normalize_model_difficulty(difficulty)
    if difficulty == "basic":
        return [
            "Review definitions and examples",
            "Solve 5 practice questions",
            "Summarize in your own words",
        ]
    return [
        "Revise core concepts",
        "Solve mixed numerical problems",
        "Attempt previous exam questions",
        "Identify weak subtopics",
    ]


def generate_motivation_message(topic, struggle_text, quiz_result):
    struggle_text = (struggle_text or "").strip()
    if quiz_result["score"] >= 80:
        tone = "You are doing well. Keep the momentum."
    elif quiz_result["score"] >= 55:
        tone = "You are close to a strong hold. Tighten the weak section."
    else:
        tone = "Progress is still possible with focused revision and short practice cycles."

    if struggle_text:
        return f"You mentioned difficulty with {struggle_text}. {tone} Stay consistent in {topic}."
    return f"{tone} Stay consistent in {topic}."


def _sentence_split(text):
    parts = re.split(r"(?<=[.!?])\s+", str(text).strip())
    return [p.strip() for p in parts if p.strip()]


def _shorten(text, max_len=110):
    t = str(text).strip()
    return t if len(t) <= max_len else t[: max_len - 3].rstrip() + "..."


def _unique_options(items):
    out = []
    seen = set()
    for item in items:
        key = item.strip().lower()
        if not item.strip() or key in seen:
            continue
        seen.add(key)
        out.append(item.strip())
    return out


def _shuffle_with_correct(options, correct_option, seed_key):
    options = _unique_options(options)
    if correct_option not in options:
        options.insert(0, correct_option)
    fillers = [
        "It is mainly a historical timeline concept.",
        "It deals only with grammar rules.",
        "It is unrelated to quantitative reasoning.",
        "It is only about memorizing definitions.",
    ]
    for filler in fillers:
        if len(options) >= 4:
            break
        if filler not in options:
            options.append(filler)

    options = options[:4]
    rng = random.Random(int(hashlib.md5(seed_key.encode("utf-8")).hexdigest(), 16))
    rng.shuffle(options)
    return options, options.index(correct_option)


def build_mcq_questions(topic_row):
    topic = topic_row["topic"]
    text = topic_row["text"]
    sentences = _sentence_split(text)

    s1 = _shorten(sentences[0] if sentences else f"{topic} is an important concept.")
    s2 = _shorten(sentences[1] if len(sentences) > 1 else f"{topic} is used in practical problem-solving.")
    s3 = _shorten(sentences[2] if len(sentences) > 2 else f"{topic} supports analytical reasoning.")

    concept_correct = s1
    concept_opts, concept_correct_idx = _shuffle_with_correct(
        [
            concept_correct,
            f"{topic.title()} focuses only on storytelling and chronology.",
            f"{topic.title()} avoids models, equations, and structured reasoning.",
            f"{topic.title()} has no practical value in solving questions.",
        ],
        concept_correct,
        f"{topic}-concept",
    )

    app_correct = s2
    app_opts, app_correct_idx = _shuffle_with_correct(
        [
            app_correct,
            f"{topic.title()} is used only for decorative terminology.",
            f"{topic.title()} has no role in real-world applications.",
            f"{topic.title()} is useful only in unrelated language tasks.",
        ],
        app_correct,
        f"{topic}-application",
    )

    formula_match = re.search(r"([A-Za-z][A-Za-z0-9_ ]*\s*=\s*[^.,;]+)", text)
    if formula_match:
        formula_correct = _shorten(formula_match.group(1))
    else:
        law_match = re.search(
            r"\b([A-Za-z]+(?:\s+[A-Za-z]+){0,3}\s+(?:law|theorem|equation|principle))\b", text, re.I
        )
        formula_correct = law_match.group(1).strip() if law_match else s3

    formula_opts, formula_correct_idx = _shuffle_with_correct(
        [
            formula_correct,
            "Profit = Selling Price - Cost Price",
            "noun + verb + object",
            "Area = pi r^2 for every topic",
        ],
        formula_correct,
        f"{topic}-formula",
    )

    return {
        "concept": {
            "prompt": f"Which statement best defines {topic}?",
            "options": concept_opts,
            "correct": concept_correct_idx,
        },
        "application": {
            "prompt": f"Which option is a valid application of {topic}?",
            "options": app_opts,
            "correct": app_correct_idx,
        },
        "formula": {
            "prompt": f"Which expression/law is most relevant to {topic}?",
            "options": formula_opts,
            "correct": formula_correct_idx,
        },
    }


def evaluate_quiz(topic_row, selected_answers):
    questions = build_mcq_questions(topic_row)

    section_scores = {}
    correct_count = 0
    mapping = [("concept", "Concept"), ("application", "Application"), ("formula", "Formula")]
    for key, label in mapping:
        picked = selected_answers.get(key, "")
        correct_idx = questions[key]["correct"]
        is_correct = str(picked).isdigit() and int(picked) == int(correct_idx)
        section_scores[label] = 1.0 if is_correct else 0.0
        if is_correct:
            correct_count += 1

    total = round((correct_count / 3.0) * 100)
    weak_section = min(section_scores, key=section_scores.get)

    if total >= 80:
        feedback = "Strong understanding. Keep practicing mixed-level questions."
    elif total >= 55:
        feedback = "Good progress, but some concepts need clearer explanation."
    else:
        feedback = "Foundational understanding is weak. Revise summary and core terms first."

    tips = generate_study_tips(topic_row["text"])
    return {
        "score": total,
        "weakness": weak_section,
        "feedback": feedback,
        "tips": tips,
    }


@app.route("/")
def home():
    subjects = sorted(DATA["subject"].unique(), key=lambda s: s.lower())
    return render_template("index.html", subjects=subjects)


@app.route("/subject/<subject_name>")
def subject_topics(subject_name):
    filtered = DATA[DATA["subject"].str.lower() == subject_name.lower()]
    if filtered.empty:
        abort(404)
    topics = filtered.to_dict(orient="records")
    return render_template("subject.html", subject_name=filtered.iloc[0]["subject"], topics=topics)


@app.route("/topic/<topic_name>")
def topic_page(topic_name):
    topic_row = TOPIC_LOOKUP.get(topic_name.lower())
    if topic_row is None:
        abort(404)

    predicted = normalize_model_difficulty(predict_difficulty(topic_row["text"]))
    study_plan = generate_study_plan(predicted)
    questions = build_mcq_questions(topic_row)

    return render_template(
        "topic.html",
        topic=topic_row.to_dict(),
        predicted_difficulty=predicted.title(),
        study_plan=study_plan,
        questions=questions,
        quiz_result=None,
        submitted_answers={},
        struggle_input="",
        motivation_message=None,
    )


@app.route("/submit_quiz", methods=["POST"])
def submit_quiz():
    topic_name = request.form.get("topic_name", "").strip()
    topic_row = TOPIC_LOOKUP.get(topic_name.lower())
    if topic_row is None:
        abort(404)

    selected_answers = {
        "concept": request.form.get("answer_concept", ""),
        "application": request.form.get("answer_application", ""),
        "formula": request.form.get("answer_formula", ""),
    }
    struggle_input = request.form.get("struggle_input", "")

    predicted = normalize_model_difficulty(predict_difficulty(topic_row["text"]))
    study_plan = generate_study_plan(predicted)
    questions = build_mcq_questions(topic_row)
    quiz_result = evaluate_quiz(topic_row, selected_answers)
    motivation_message = generate_motivation_message(topic_row["topic"], struggle_input, quiz_result)

    return render_template(
        "topic.html",
        topic=topic_row.to_dict(),
        predicted_difficulty=predicted.title(),
        study_plan=study_plan,
        questions=questions,
        quiz_result=quiz_result,
        submitted_answers=selected_answers,
        struggle_input=struggle_input,
        motivation_message=motivation_message,
    )


if __name__ == "__main__":
    app.run(debug=True)
