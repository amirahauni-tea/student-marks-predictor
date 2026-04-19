import streamlit as st
import numpy as np
import pickle
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Student Marks Predictor",page_icon="🔖",layout="centered")

st.title("👩‍🎓 Student Marks Predictor")
st.write("Enter The Number of Hours Studied ⌚️ (1-10) And **Click Predict** To See The Predicted Marks")

# Load the Model
def load_model(model):
    with open(model,"rb") as f:
        slr = pickle.load(f)
    return slr

try:
    model = load_model("simple_linear_regression.pkl")
except Exception as e:
    st.error("Your Pickle File Not Found")
    st.exception(e)
    st.stop()

hours = st.number_input("Hours  Studied",
                        min_value=1.0,
                        max_value=10.0,
                        value=4.0,
                        step=0.1,
                        format="%.1f")

if st.button("Predict"):
    try:
        X = np.array([[hours]])
        predictions = model.predict(X)
        predictions = predictions[0]  # getting answer not in array form
        st.success(f"✅ Predicted Marks : {predictions:.2f}")
        st.write("⛔️ Note : This is ML Model Prediction **Result May Vary**")
    except Exception as e:
        st.error(f"Prediction Failed : {e}")
