# streamlit_app.py

import streamlit as st
import time
import os
import re
import numpy as np
from typing import List, Dict, Optional

# --- Machine Learning and NLTK imports ---
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# ===================================================================
# 1. PAGE CONFIGURATION & BOT IDENTITY
# ===================================================================

# Set the identity of the chatbot
BOT_NAME = "Chatbot Genie"
BOT_AVATAR = "🧞‍♂️"
USER_AVATAR = "👤"

st.set_page_config(
    page_title=BOT_NAME,
    page_icon="🧞‍♂️",
    layout="centered",
    initial_sidebar_state="auto"
)

# ===================================================================
# 2. CUSTOM CSS FOR BRIGHT STYLING (UPDATED COLORS)
# ===================================================================

def load_css():
    """Inject custom CSS for a bright and friendly theme."""
    st.markdown("""
        <style>
            /* Chat bubble styles */
            .stChatMessage {
                border-radius: 20px;
                padding: 1rem 1.25rem;
                margin-bottom: 1rem;
                box-shadow: 0 4px 6px rgba(0,0,0,0.05);
                border: 1px solid rgba(0,0,0,0.05);
                transition: all 0.3s ease;
            }

            /* User message (aligned right) - UPDATED to bright coral */
            div[data-testid="stChatMessage"]:has(div[data-testid="stAvatarIcon-user"]) {
                background-color: #FF6F61; /* Corresponds to primaryColor in theme */
                color: white;
            }
            
            /* Ensure text inside user message is white */
            div[data-testid="stChatMessage"]:has(div[data-testid="stAvatarIcon-user"]) p {
                color: white;
            }

            /* Bot message (aligned left) */
            div[data-testid="stChatMessage"]:has(div[data-testid="stAvatarIcon-assistant"]) {
                background-color: #FFFFFF; /* Corresponds to secondaryBackgroundColor */
            }

            /* Buttons styling - UPDATED to bright coral */
            .stButton>button {
                border-radius: 20px;
                border: 1px solid #FF6F61;
            }
        </style>
    """, unsafe_allow_html=True)

load_css()

# ===================================================================
# 3. CHATBOT LOGIC (Unchanged)
# ===================================================================

# --- Configuration Constants ---
DATA_FILEPATH = "Payroll_Queries.txt"
CONFIDENCE_THRESHOLD = 0.2

@st.cache_resource
def load_nltk_data():
    try: nltk.data.find('corpora/stopwords')
    except LookupError: nltk.download('stopwords', quiet=True)
    try: nltk.data.find('corpora/wordnet.zip')
    except LookupError: nltk.download('wordnet', quiet=True)

load_nltk_data()

STOP_WORDS = set(stopwords.words('english'))
LEMMATIZER = WordNetLemmatizer()

def preprocess_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    words = [LEMMATIZER.lemmatize(word) for word in text.split() if word not in STOP_WORDS]
    return ' '.join(words)

@st.cache_resource
def load_payroll_data_from_file(filepath: str = DATA_FILEPATH) -> Optional[List[Dict[str, str]]]:
    if not os.path.exists(filepath): return None
    data = []
    qa_pattern = re.compile(r"Question:\s*(.*?)\s*Answer:\s*(.*?)(?=Question:|$)", re.DOTALL | re.IGNORECASE)
    try:
        with open(filepath, 'r', encoding='utf-8') as file:
            content = file.read()
            matches = qa_pattern.findall(content)
            for q_raw, a_raw in matches:
                if q_raw.strip() and a_raw.strip():
                    data.append({'raw_question': q_raw.strip(), 'question': preprocess_text(q_raw.strip()), 'answer': a_raw.strip()})
    except Exception: return None
    return data if data else None

class PayrollChatbot:
    def __init__(self, data: List[Dict[str, str]]):
        if not data: raise ValueError("Data cannot be empty.")
        self.data = data
        self.processed_questions = [item['question'] for item in self.data]
        if not any(self.processed_questions): raise ValueError("No valid training data.")
        self.vectorizer = TfidfVectorizer()
        self.question_vectors = self.vectorizer.fit_transform(self.processed_questions)

    def find_best_match(self, user_query: str) -> Optional[Dict]:
        if not user_query.strip(): return None
        processed_query = preprocess_text(user_query)
        if not processed_query: return None
        query_vector = self.vectorizer.transform([processed_query])
        similarities = cosine_similarity(query_vector, self.question_vectors)
        most_similar_index = np.argmax(similarities)
        max_similarity = similarities[0, most_similar_index]
        if max_similarity > CONFIDENCE_THRESHOLD:
            match = self.data[most_similar_index]
            return {"matched_question": match['raw_question'], "answer": match['answer'], "score": max_similarity}
        return None

# ===================================================================
# 4. STREAMLIT UI & APP LOGIC (MODIFIED)
# ===================================================================

# --- Sidebar ---
with st.sidebar:
    st.title(f"🧞‍♂️ {BOT_NAME}")
    st.markdown(f"Welcome! I am {BOT_NAME}, your virtual assistant for payroll-related questions.")
    
    if st.button("Clear Chat History"):
        st.session_state.clear()
        st.rerun()

    st.markdown("---")
    st.markdown("Developed with ❤️ using [Streamlit](https://streamlit.io/)")

# --- Main Page Header & Creator Info ---
col1, col2 = st.columns([3, 1], gap="medium")
with col1:
    st.header(f"Chat with {BOT_NAME}")

with col2:
    st.image("Image.jpg", caption="CA Shiv Madaan", width=120) 

# --- Chat Logic ---
@st.cache_resource
def initialize_chatbot():
    payroll_data = load_payroll_data_from_file()
    return PayrollChatbot(payroll_data) if payroll_data else None

chatbot = initialize_chatbot()

if not chatbot:
    st.error("Fatal Error: Chatbot could not be initialized. Please check 'Payroll_Queries.txt'.")
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": f"Hello, how are you? I am {BOT_NAME}, your virtual payroll assistant. How can I help you today?"}]
if "pending_confirmation" not in st.session_state:
    st.session_state.pending_confirmation = None

def add_message(role, content, avatar):
    st.session_state.messages.append({"role": role, "content": content})
    with st.chat_message(role, avatar=avatar):
        st.markdown(content)

for message in st.session_state.messages:
    with st.chat_message(message["role"], avatar=USER_AVATAR if message["role"] == "user" else BOT_AVATAR):
        st.markdown(message["content"])

if st.session_state.pending_confirmation:
    match_data = st.session_state.pending_confirmation
    col1, col2 = st.columns(2)
    with col1:
        if st.button("✅ Yes, that's correct", use_container_width=True):
            with st.chat_message("assistant", avatar=BOT_AVATAR):
                with st.spinner("Finding the answer..."):
                    time.sleep(1.5)
                    st.markdown(match_data["answer"])
            st.session_state.messages.append({"role": "assistant", "content": match_data["answer"]})
            st.session_state.pending_confirmation = None
            st.rerun()
    with col2:
        if st.button("❌ No, that's not right", use_container_width=True):
            with st.chat_message("assistant", avatar=BOT_AVATAR):
                with st.spinner("Thinking..."):
                    time.sleep(1)
                    st.markdown("My apologies. Could you please rephrase your question?")
            st.session_state.messages.append({"role": "assistant", "content": "My apologies. Could you please rephrase your question?"})
            st.session_state.pending_confirmation = None
            st.rerun()

elif prompt := st.chat_input("Ask your question here..."):
    add_message("user", prompt, USER_AVATAR)
    match = chatbot.find_best_match(prompt)
    with st.chat_message("assistant", avatar=BOT_AVATAR):
        with st.spinner("Analyzing your question..."):
            time.sleep(2)
            if match:
                confirmation_question = f"I found a related question: **'{match['matched_question']}'**. Is this what you were asking about?"
                st.markdown(confirmation_question)
                st.session_state.messages.append({"role": "assistant", "content": confirmation_question})
                st.session_state.pending_confirmation = match
                st.rerun()
            else:
                no_match_response = "I'm sorry, I couldn't find a specific answer for that. Please try rephrasing or contact HR for more assistance."
                st.markdown(no_match_response)
                st.session_state.messages.append({"role": "assistant", "content": no_match_response})

st.html("""
<script>
    const inputs = window.parent.document.querySelectorAll("textarea[data-testid='stChatInput']");
    if (inputs.length > 0) {
        inputs[0].focus();
    }
</script>
""")