import streamlit as st
import pandas as pd
import joblib

@st.cache_resource
def load_model():
    return joblib.load('titanic_pipeline.pkl')

model = load_model()

st.set_page_config(
    page_title = "Titanic Survival Prediction",
    page_icon = "🚢",
    layout = "centered",
)

st.title("🚢 Titanic Survival Prediction")
st.write(
    "Enter passenger details to predict "
    "whether they would survived the Titanic disaster. "
)

st.divider()

st.subheader("Passenger Information")

col1, col2 = st.columns(2)

with col1:
    pclass = st.selectbox(
        "Passenger Class", 
        [1, 2, 3],
        index=2
    )

    sex = st.selectbox(
        "Sex",
        ["Male", "Female"]
    )

    age = st.number_input(
        "Age",
        min_value=0.0,
        max_value=100.0,
        value=25.0,
        step=1.0
    )

    sibsp = st.number_input(
        "Siblings/Spouses Aboard",
        min_value=0,
        max_value=10,
        value=0,
    )

with col2:
    parch = st.number_input(
        "Parents/ Children Aboard",
        min_value=0,
        max_value=10,
        value=0,
    )

    fare = st.number_input(
        "Ticket Fare",
        min_value=0.0,
        max_value=600.0,
        value=7.25,
        step=0.5
    )

    embarked = st.selectbox(
        "Port of Embarkation",
        ["S", "C", "Q"]
    )

# Prediction
if st.button("Predict Survival", type="primary",
             use_container_width=True):

    passenger = pd.DataFrame([{
        "Pclass": pclass,
        "Sex": sex,
        "Age": age,
        "SibSp": sibsp,
        "Parch": parch,
        "Fare": fare,
        "Embarked": embarked
    }])

    prediction = model.predict(passenger)[0]
    probabilities = model.predict_proba(passenger)[0]

    st.divider()
    st.subheader("Prediction Result")

    if prediction == 1:
        st.success("Prediction: Passenger Survived! 🎉")
    else:
        st.error("Prediction: Passenger Did Not Survive.")

    st.metric(
        "Estimated Survival Probability",
        f"{probabilities[1] * 100:.2f}%"
    )

    st.caption(
        "This is a machine learning estimate, "
        "not a guarantee of the historical outcome."
    )
