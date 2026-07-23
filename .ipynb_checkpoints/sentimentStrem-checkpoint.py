import streamlit as st
import joblib

# ----------------------------
# Page Config
# ----------------------------
st.set_page_config(
    page_title="Sentiment Analysis AI",
    page_icon="😊",
    layout="wide"
)

# ----------------------------
# Load Model
# ----------------------------
model = joblib.load("model.pkl")
vectorizer = joblib.load("vect.pkl")

# ----------------------------
# Custom CSS
# ----------------------------
st.markdown("""
<style>

.main{
    background-color:#f5f7fa;
}

.title{
    text-align:center;
    font-size:42px;
    font-weight:bold;
    color:#4F46E5;
}

.subtitle{
    text-align:center;
    color:gray;
    font-size:18px;
}

.result{
    padding:20px;
    border-radius:15px;
    text-align:center;
    font-size:30px;
    font-weight:bold;
}

</style>
""", unsafe_allow_html=True)

# ----------------------------
# Heading
# ----------------------------

st.markdown("<div class='title'>😊 AI Sentiment Analysis</div>", unsafe_allow_html=True)

st.markdown("<div class='subtitle'>Analyze whether a sentence is Positive or Negative using Machine Learning</div>", unsafe_allow_html=True)

st.write("")

# ----------------------------
# Text Input
# ----------------------------

text = st.text_area(
    "✍ Enter your text",
    height=180,
    placeholder="Example : I really love this product."
)

# ----------------------------
# Predict Button
# ----------------------------

if st.button("🔍 Analyze Sentiment", use_container_width=True):

    if text == "":
        st.warning("Please enter some text.")
    else:

        vector = vectorizer.transform([text])

        prediction = model.predict(vector)[0]

        st.write("")

        if prediction.lower() == "positive":

            st.success("😊 Positive Sentiment")

            st.balloons()

        elif prediction.lower() == "negative":

            st.error("😞 Negative Sentiment")

        else:

            st.info(prediction)

# ----------------------------
# Sidebar
# ----------------------------

st.sidebar.title("About")

st.sidebar.write("""
This project uses

✅ TF-IDF Vectorizer

✅ Logistic Regression

✅ Scikit-Learn

✅ Streamlit
""")

st.sidebar.success("Model Ready")

st.sidebar.info("Created by Muskan Yadav")