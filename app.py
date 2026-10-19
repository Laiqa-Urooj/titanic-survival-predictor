import pickle
from pathlib import Path

import pandas as pd
import streamlit as st

# Load the SVC model trained in the Titanic notebook.
MODEL_PATH = Path(__file__).parent / "svc_trained_model.pkl"

st.set_page_config(page_title="Titanic Survival Predictor", page_icon="🚢")
st.title("🚢 Titanic Passenger Survival Predictor")
st.write(
    "Enter the passenger details below. The model predicts survival from "
    "four features: passenger class, gender, siblings, and port of embarkation."
)

# These categories are fitted using sklearn LabelEncoder, which sorts classes
# alphabetically. They must match the notebook's encoders exactly.
ENCODINGS = {
    "PClass": {"First": 0, "Second": 1, "Third": 2},
    "Gender": {"Female": 0, "Male": 1},
    "Sibling": {"One": 0, "Three": 1, "Two": 2, "Zero": 3},
    "Embarked": {"Cherbourg": 0, "Queenstown": 1, "Southampton": 2},
}

@st.cache_resource
def load_model():
    with MODEL_PATH.open("rb") as file:
        return pickle.load(file)

pclass = st.selectbox("Passenger class", ["First", "Second", "Third"])
gender = st.selectbox("Gender", ["Female", "Male"])
sibling = st.selectbox("Number of siblings", ["Zero", "One", "Two", "Three"])
embarked = st.selectbox(
    "Port of embarkation", ["Cherbourg", "Queenstown", "Southampton"]
)

if st.button("Predict survival", type="primary"):
    encoded_input = pd.DataFrame([{
        "PClass": ENCODINGS["PClass"][pclass],
        "Gender": ENCODINGS["Gender"][gender],
        "Sibling": ENCODINGS["Sibling"][sibling],
        "Embarked": ENCODINGS["Embarked"][embarked],
    }], columns=["PClass", "Gender", "Sibling", "Embarked"])

    try:
        model = load_model()
        prediction = int(model.predict(encoded_input)[0])
        # Notebook target encoder uses LabelEncoder on ["Yes", "No"]:
        # No -> 0, Yes -> 1.
        if prediction == 1:
            st.success("Prediction: SURVIVED")
        else:
            st.error("Prediction: DID NOT SURVIVE")
        st.caption(
            "This is a machine-learning estimate based on the notebook's "
            "training data, not a guarantee."
        )
    except Exception as exc:
        st.error("The model could not make a prediction. Check the deployment logs.")
        st.exception(exc)

st.divider()
st.caption("Model: Support Vector Classifier (SVC) from the accompanying notebook.")
