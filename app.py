import streamlit as st
import joblib
import time
import pandas as pd
import random
import spacy

# Page Config
st.set_page_config(page_title="ReviewRadar | Live Monitor", layout="wide")

# --- LOAD RESOURCES ---
@st.cache_resource
def load_resources():
    # Load Sentiment Model
    model = joblib.load('sentiment_model.pkl')
    # Load NLP Model for NER (Small English model)
    nlp = spacy.load("en_core_web_sm")
    return model, nlp

model, nlp = load_resources()

# Mock Data
MOCK_REVIEWS = [
    "The iPhone 15 battery life is amazing, truly a game changer for Apple.",
    "Terrible customer service at Walmart, the item arrived broken on Monday.",
    "Netflix subscription is too expensive now in the USA.",
    "Absolutely love this! Best purchase of 2024.",
    "Stopped working after two days. Refund requested from Amazon.",
    "Fast shipping by DHL and great quality packaging.",
    "The Sony headphones interface is clunky and confusing.",
    "Five stars! Highly recommended for anyone in London.",
]

# --- UI Layout ---
st.title("Real-Time Review Analysis")

# Sidebar
st.sidebar.header("Control Panel")
app_mode = st.sidebar.radio("Select Mode", ["Manual Analysis", "Live Stream Simulator"])

# --- MODE 1: Manual Analysis ---
if app_mode == "Manual Analysis":
    st.subheader("Review Analysis")
    user_input = st.text_area("Paste a review here to analyze:", height=150)
    
    if st.button("Analyze Review"):
        if user_input:
            # 1. Sentiment Prediction
            prediction = model.predict([user_input])[0]
            probability = model.predict_proba([user_input]).max()
            
            # 2. NER Extraction (The Upgrade)
            doc = nlp(user_input)
            entities = [(ent.text, ent.label_) for ent in doc.ents]
            
            # --- Display Results ---
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### Sentiment")
                if prediction == 1:
                    st.success(f"**Positive** ({probability:.2%})")
                else:
                    st.error(f"**Negative** ({probability:.2%})")
            
            with col2:
                st.markdown("### Named Entities Detected")
                if entities:
                    for text, label in entities:
                        # Display as colorful tags
                        st.markdown(f"**{text}** : `{label}`")
                else:
                    st.info("No specific entities found (e.g., Brands, Dates, Places).")

# --- MODE 2: Live Stream Simulator ---
elif app_mode == "Live Stream Simulator":
    st.subheader("📡 Live Feed Monitor")
    
    if 'data_log' not in st.session_state:
        st.session_state.data_log = pd.DataFrame(columns=["Review", "Sentiment", "Entities"])

    start_btn = st.button("Start/Stop Stream")
    
    col1, col2 = st.columns([2, 1])
    with col1:
        feed_holder = st.empty()
    with col2:
        metrics_holder = st.empty()

    if start_btn:
        for _ in range(10):
            new_review = random.choice(MOCK_REVIEWS)
            
            # Sentiment
            pred = model.predict([new_review])[0]
            label = "Positive" if pred == 1 else "Negative"
            
            # NER
            doc = nlp(new_review)
            # Format entities as a string for the table (e.g., "Apple (ORG), Monday (DATE)")
            ents_list = [f"{ent.text} ({ent.label_})" for ent in doc.ents]
            ents_str = ", ".join(ents_list) if ents_list else "-"
            
            # Update Log
            new_row = pd.DataFrame({"Review": [new_review], "Sentiment": [label], "Entities": [ents_str]})
            st.session_state.data_log = pd.concat([new_row, st.session_state.data_log], ignore_index=True)
            
            # Display Live Feed (Table View for NER clarity)
            with feed_holder:
                # Show top 5 rows, styling the Sentiment column
                display_df = st.session_state.data_log.head(5)
                st.dataframe(display_df, use_container_width=True, hide_index=True)

            # Metrics
            with metrics_holder:
                total = len(st.session_state.data_log)
                pos_pct = len(st.session_state.data_log[st.session_state.data_log['Sentiment']=='Positive']) / total
                st.metric("Live Positive Rate", f"{pos_pct:.0%}")
                st.metric("Total Processed", total)
            
            time.sleep(2)