import streamlit as st
import pandas as pd
import requests
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="E-Commerce AI Assistant", page_icon="🛍️", layout="wide")
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3.2"

@st.cache_data
def load_data():
    return (pd.read_csv("data/products.csv"),
            pd.read_csv("data/faqs.csv"),
            pd.read_csv("data/reviews.csv"))

@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

products, faqs, reviews = load_data()
with st.spinner("Loading AI model..."):
    model = load_model()

def ask_ollama(prompt):
    try:
        r = requests.post(OLLAMA_URL, json={
            "model": OLLAMA_MODEL, "prompt": prompt, "stream": False
        }, timeout=120)
        r.raise_for_status()
        return r.json().get("response", "No response received.")
    except requests.exceptions.ConnectionError:
        return "Cannot connect to Ollama. Make sure Ollama is running."
    except requests.exceptions.RequestException as e:
        return f"Ollama error: {e}"

def similar(query, texts):
    if not texts:
        return []
    q = model.encode([query])
    vectors = model.encode(texts)
    return cosine_similarity(q, vectors)[0].argsort()[::-1]

def detect_intent(query):
    prompt = f"""Classify this customer message into exactly one category:
FAQ, RECOMMENDATION, PRODUCT, REVIEW, UNKNOWN.
FAQ means delivery/return/payment/warranty.
RECOMMENDATION means asking for product suggestions.
PRODUCT means asking about a specific product's details or price.
REVIEW means asking about customer opinions or reviews.
UNKNOWN means anything else.
Message: {query}
Return only the category."""
    result = ask_ollama(prompt).upper()
    for label in ["RECOMMENDATION", "PRODUCT", "REVIEW", "FAQ", "UNKNOWN"]:
        if label in result:
            return label
    return "UNKNOWN"

def answer_faq(query):
    questions = faqs.question.astype(str).tolist()
    idx = similar(query, questions)[0]
    prompt = f"""Answer using only this FAQ. Do not invent information.
FAQ: {questions[idx]}
Answer: {faqs.iloc[idx]['answer']}
Customer question: {query}
If the FAQ does not answer it, say the information is unavailable."""
    return ask_ollama(prompt)

def recommend(query):
    descriptions = (products.name.astype(str) + " " +
                    products.category.astype(str) + " " +
                    products.description.astype(str)).tolist()
    indexes = similar(query, descriptions)[:3]
    rows = products.iloc[indexes]
    return "### Recommended products\n" + "\n\n".join(
        f"**{r['name']}** — Rs. {r['price']}  \n{r['description']}"
        for _, r in rows.iterrows()
    )

def answer_product(query):
    descriptions = (products.name.astype(str) + " " +
                    products.category.astype(str) + " " +
                    products.description.astype(str)).tolist()
    idx = similar(query, descriptions)[0]
    p = products.iloc[idx]
    prompt = f"""Answer only from these product details. Do not invent features.
Name: {p['name']}
Category: {p['category']}
Price: Rs. {p['price']}
Description: {p['description']}
Customer question: {query}
If the answer is not in the details, say so."""
    return ask_ollama(prompt)

def summarize_reviews(query):
    names = products.name.astype(str).tolist()
    match = next((n for n in names if n.lower() in query.lower()), None)
    if match is None:
        match = names[similar(query, names)[0]]
    items = reviews.loc[reviews.product_name.str.lower() == match.lower(), "review"].dropna().tolist()
    if not items:
        return f"No reviews available for {match}."
    return ask_ollama(
        f"Summarize these reviews for {match}. Mention positive and negative points. "
        f"Do not add unsupported information.\nReviews:\n" + "\n".join(items)
    )

def process_query(query):
    intent = detect_intent(query)
    if intent == "FAQ":
        answer = answer_faq(query)
    elif intent == "RECOMMENDATION":
        answer = recommend(query)
    elif intent == "PRODUCT":
        answer = answer_product(query)
    elif intent == "REVIEW":
        answer = summarize_reviews(query)
    else:
        answer = "I can help with FAQs, product details, recommendations and customer reviews."
    return intent, answer

# Sidebar dashboard
with st.sidebar:
    st.subheader("📊 Dataset Information")
    st.metric("Products", len(products))
    st.metric("FAQs", len(faqs))
    st.metric("Reviews", len(reviews))
    st.divider()
    st.subheader("Capabilities")
    for item in ["FAQ Search", "Product Recommendation", "Product Q&A",
                 "Review Analysis", "Ollama LLM", "Intent Detection"]:
        st.markdown("✅ " + item)
    st.divider()
    st.caption("Powered by RAG, Sentence Transformers and Ollama")

st.markdown("<h1 style='text-align:center'>🛍️ E-Commerce AI Assistant</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;color:gray'>Ask about products, orders, recommendations and reviews</p>", unsafe_allow_html=True)

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("intent"):
            st.caption("Detected intent: " + msg["intent"])

query = st.chat_input("Ask me anything about our products...")
if query:
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)
    with st.chat_message("assistant"):
        with st.spinner("Generating response..."):
            intent, answer = process_query(query)
        st.markdown(answer)
        st.caption("Detected intent: " + intent)
    st.session_state.messages.append({
        "role": "assistant", "content": answer, "intent": intent
    })

with st.expander("Try sample questions"):
    st.markdown(
        "- What is your return policy?\n"
        "- Recommend a good laptop\n"
        "- What is the price of Samsung Galaxy S24?\n"
        "- Summarize Samsung Galaxy S24 reviews"
    )
