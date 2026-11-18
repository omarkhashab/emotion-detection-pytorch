"""
EmotionSense - Streamlit interface (Section 4).

Run:
    streamlit run app.py

How Streamlit works (important for understanding this file):
    Streamlit re-runs this whole script from top to bottom on every interaction
    (button click, text edit). Two tools keep that cheap and stateful:
      * @st.cache_resource -> the model is loaded ONCE per process, not per click.
      * st.session_state   -> remembers the text box contents and last prediction
                              between re-runs.

All preprocessing and inference live in src/predict.py; this file is only UI.
"""

from __future__ import annotations

import altair as alt
import pandas as pd
import streamlit as st

from src.predict import EmotionClassifier, Prediction

EMOJI = {"sadness": "😢", "joy": "😄", "love": "❤️", "anger": "😠", "fear": "😨", "surprise": "😲"}
EXAMPLES = [
    "I am so excited about tomorrow!",
    "I feel so alone since my best friend moved away.",
    "I can't believe they cancelled my flight again without telling me.",
    "I'm really nervous about walking home alone tonight.",
    "I was shocked to see my whole family waiting at the airport.",
]
HIGHLIGHT, MUTED_BAR = "#2a78d6", "#b7d3f6"  # predicted class vs. the others

st.set_page_config(page_title="EmotionSense", page_icon="🎭", layout="centered")


# --------------------------------------------------------------------------- #
# Model loading - cached for the lifetime of the Streamlit server process
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner="Loading the emotion model ...")
def load_classifier() -> EmotionClassifier:
    return EmotionClassifier()


# --------------------------------------------------------------------------- #
# Callbacks (run before the script re-executes)
# --------------------------------------------------------------------------- #
def use_example(text: str) -> None:
    st.session_state.text = text
    st.session_state.pop("result", None)  # clear the old prediction


def probability_chart(result: Prediction) -> alt.Chart:
    """Horizontal bar chart of all six probabilities, sorted high -> low."""
    df = pd.DataFrame({
        "emotion": [label.capitalize() for label in result.probabilities],
        "probability": list(result.probabilities.values()),
    })
    is_top = alt.datum.emotion == result.label.capitalize()
    base = alt.Chart(df).encode(
        y=alt.Y("emotion:N", sort="-x", title=None),
        x=alt.X("probability:Q", title="Probability", scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format="%")),
        tooltip=[alt.Tooltip("emotion:N", title="Emotion"), alt.Tooltip("probability:Q", title="Probability", format=".2%")],
    )
    bars = base.mark_bar(cornerRadiusEnd=4, height=18).encode(
        color=alt.condition(is_top, alt.value(HIGHLIGHT), alt.value(MUTED_BAR))
    )
    labels = base.mark_text(align="left", dx=4).encode(text=alt.Text("probability:Q", format=".1%"))
    return (bars + labels).properties(height=230)


# --------------------------------------------------------------------------- #
# Layout
# --------------------------------------------------------------------------- #
st.title("🎭 EmotionSense")
st.caption("Emotion detection with a fine-tuned DistilBERT - sadness, joy, love, anger, fear, surprise.")

try:
    classifier = load_classifier()
except FileNotFoundError as err:
    st.error(str(err))
    st.stop()

st.session_state.setdefault("text", "")

st.text_area(
    "Enter a sentence to analyze:",
    key="text",
    height=110,
    max_chars=1000,
    placeholder="e.g. I am so excited about tomorrow!",
)

if st.button("Predict emotion", type="primary"):
    text = st.session_state.text
    st.session_state.result = None
    if not text.strip():
        st.warning("Please enter a sentence before predicting.")  # reject empty input
    else:
        with st.spinner("Analyzing ..."):
            try:
                st.session_state.result = classifier.predict(text)
            except Exception as err:  # surface inference problems instead of crashing
                st.error(f"Prediction failed: {err}")

with st.expander("Try an example sentence", expanded=False):
    for i, example in enumerate(EXAMPLES):
        st.button(example, key=f"example_{i}", on_click=use_example, args=(example,))

result: Prediction | None = st.session_state.get("result")
if result is not None:
    st.divider()
    left, right = st.columns([1, 2])
    with left:
        st.metric("Prediction", f"{EMOJI.get(result.label, '')} {result.label.upper()}")
        st.metric("Confidence", f"{result.confidence:.1%}")
    with right:
        st.markdown("**All six class probabilities**")
        st.altair_chart(probability_chart(result))

st.divider()
st.caption(
    "Confidence is the model's softmax probability, not a guarantee of correctness. "
    "This is an educational text classifier - not a clinical or mental-health tool."
)
