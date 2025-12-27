import streamlit as st
import joblib
import time
import pandas as pd
import spacy
import feedparser

# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------
st.set_page_config(
    page_title="ReviewRadar | Live News Monitor",
    layout="wide"
)

# -------------------------------------------------
# LOAD MODELS
# -------------------------------------------------
@st.cache_resource
def load_resources():
    model = joblib.load("sentiment_model.pkl")
    nlp = spacy.load("en_core_web_sm")
    return model, nlp

model, nlp = load_resources()

# -------------------------------------------------
# GOOGLE NEWS RSS FETCHER
# -------------------------------------------------
def fetch_news(query="technology", max_items=5):
    query = query.replace(" ", "+")
    url = (
        f"https://news.google.com/rss/search?"
        f"q={query}&hl=en-IN&gl=IN&ceid=IN:en"
    )

    feed = feedparser.parse(url)
    articles = []

    for entry in feed.entries[:max_items]:
        text = f"{entry.title}. {entry.get('summary', '')}"
        articles.append(text)

    return articles

# -------------------------------------------------
# UI
# -------------------------------------------------
st.title("Real-Time Sentiment & NER Dashboard (Google News RSS)")

st.sidebar.header("Control Panel")
app_mode = st.sidebar.radio(
    "Select Mode",
    ["Manual Analysis", "Live News Stream"]
)

# -------------------------------------------------
# MODE 1: MANUAL ANALYSIS
# -------------------------------------------------
if app_mode == "Manual Analysis":
    st.subheader("Manual Text Analysis")

    user_input = st.text_area(
        "Paste a review or news text:",
        height=180
    )

    if st.button("Analyze Text"):
        if user_input.strip():
            prediction = model.predict([user_input])[0]
            probability = model.predict_proba([user_input]).max()

            doc = nlp(user_input)
            entities = [(ent.text, ent.label_) for ent in doc.ents]

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("### Sentiment")
                if prediction == 1:
                    st.success(f"Positive ({probability:.2%})")
                else:
                    st.error(f"Negative ({probability:.2%})")

            with col2:
                st.markdown("### Named Entities")
                if entities:
                    for text, label in entities:
                        st.markdown(f"**{text}** : `{label}`")
                else:
                    st.info("No named entities detected.")

# -------------------------------------------------
# MODE 2: LIVE GOOGLE NEWS STREAM
# -------------------------------------------------
else:
    st.subheader("Live Google News Stream")

    keyword = st.text_input(
        "Search keyword",
        value="technology"
    )

    refresh_rate = st.slider(
        "Refresh interval (seconds)",
        min_value=10,
        max_value=60,
        value=20
    )

    if "data_log" not in st.session_state:
        st.session_state.data_log = pd.DataFrame(
            columns=["Live News Content", "Sentiment", "Named Entities"]
        )

    start = st.button("Start Stream")

    feed_holder = st.empty()
    metrics_holder = st.empty()

    if start:
        while True:
            news_items = fetch_news(keyword, max_items=5)

            for text in news_items:
                pred = model.predict([text])[0]
                label = "Positive" if pred == 1 else "Negative"

                doc = nlp(text)
                ents = [f"{e.text} ({e.label_})" for e in doc.ents]
                ents_str = ", ".join(ents) if ents else "-"

                new_row = pd.DataFrame({
                    "Live News Content": [text],
                    "Sentiment": [label],
                    "Named Entities": [ents_str]
                })

                st.session_state.data_log = pd.concat(
                    [new_row, st.session_state.data_log],
                    ignore_index=True
                )

            # -------------------------------
            # FIXED TABLE DISPLAY
            # -------------------------------
            display_df = st.session_state.data_log.head(20)

            styled_df = display_df.style.set_properties(**{
                "white-space": "pre-wrap",
                "text-align": "left"
            })

            with feed_holder:
                st.dataframe(
                    styled_df,
                    use_container_width=True,
                    height=650
                )

            with metrics_holder:
                total = len(st.session_state.data_log)
                pos_rate = (
                    len(
                        st.session_state.data_log[
                            st.session_state.data_log["Sentiment"] == "Positive"
                        ]
                    ) / total
                )
                st.metric("Positive Rate", f"{pos_rate:.0%}")
                st.metric("Total Articles Processed", total)

            time.sleep(refresh_rate)

# -------------------------------------------------
# DISCLAIMER
# -------------------------------------------------
st.caption(
    "This application analyzes publicly available Google News RSS content. "
    "Data is processed temporarily for research and demonstration purposes only."
)
