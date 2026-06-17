import csv
import nltk
import re
import random
import gradio as gr
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

nltk.download('punkt_tab', quiet=True)  # Use new tokenizer

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    return text

# Load CSV (make sure filename matches yours – 'Health.csv' or 'kidney_health.csv')
faq_questions = []
faq_answers = []
with open('Health.csv', newline='', encoding='utf-8') as file:
    reader = csv.reader(file)
    next(reader)
    for row in reader:
        faq_questions.append(row[0])
        faq_answers.append(row[1])

# Vectorise using cleaned questions
vectorizer = TfidfVectorizer(tokenizer=word_tokenize)
cleaned_questions = [clean_text(q) for q in faq_questions]
faq_vectors = vectorizer.fit_transform(cleaned_questions)

extra_tips = [
    "🚭 Reminder: Smoking damages kidney blood vessels – even one cigarette a day.",
    "💧 Drink water instead of soda. Your kidneys will filter better.",
    "🧂 Too much salt makes kidneys work overtime. Skip the extra pinch.",
    "💊 Painkillers are not sweets – take only when necessary.",
    "🏃‍♂️ 30 minutes of play = healthier kidneys for life.",
    "😴 Sleep 8 hours – kidneys repair at night.",
    "🍎 An apple a day keeps kidney stones away (it’s rich in good fibre)."
]

def get_best_answer(user_question):
    if not user_question or user_question.strip() == "":
        return "Please ask a question about kidney health (e.g., 'How to keep kidneys healthy?')"
    
    cleaned_user = clean_text(user_question)
    user_vector = vectorizer.transform([cleaned_user])
    similarities = cosine_similarity(user_vector, faq_vectors).flatten()
    best_idx = similarities.argmax()
    
    if similarities[best_idx] < 0.2:
        return "I don't have that answer. Please rephrase or ask something like 'What foods are good for kidneys?'"
    
    answer = faq_answers[best_idx]
    if random.random() < 0.3:
        answer += "\n\n💡 " + random.choice(extra_tips)
    return answer

def chat_function(message, history):
    return get_best_answer(message)

demo = gr.ChatInterface(
    fn=chat_function,
    title="🩺 Kidney Health Advisor Chatbot",
    description="Ask me anything about keeping your kidneys healthy – what to eat, what to avoid, causes of kidney disease, and how to prevent it.\n\nExamples: 'How to keep kidneys healthy?', 'Is smoking bad for kidneys?', 'What foods should I avoid?'",
    theme="soft",
    examples=[
        "How to keep kidneys healthy?",
        "Is smoking harmful to kidneys?",
        "What foods are bad for kidneys?",
        "Can painkillers damage kidneys?",
        "How can I prevent kidney disease?",
        "My friend smokes – what should I tell him?"
    ],
    cache_examples=False
)

if __name__ == "__main__":
    demo.launch()
