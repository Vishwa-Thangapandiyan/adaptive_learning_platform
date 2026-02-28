def generate_study_plan(topic, difficulty):
    return f"Study plan for {topic} at {difficulty} level."

def generate_quiz(topic):
    return [
        f"What is {topic}?",
        f"Explain key concepts of {topic}."
    ]

def extract_keywords(text):
    words = text.split()
    return list(set(words[:5]))

def generate_summary(text):
    return text[:200] + "..."