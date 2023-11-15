import pandas as pd
import joblib
import streamlit as st

# Function to load the trained pipeline
def load_pipeline(pipeline_path):
    return joblib.load(pipeline_path)

# Streamlit app
def main():
    st.title("Math Score Prediction")

    # Creating a form for user input
    with st.form("prediction_form"):
        gender = st.selectbox("Gender", options=["male", "female"])
        race_ethnicity = st.selectbox("Race/Ethnicity", options=["group A", "group B", "group C", "group D", "group E"])
        parental_level_of_education = st.selectbox("Parental Level of Education", options=["bachelor's degree", "some college", "master's degree", "associate's degree", "high school", "some high school"])
        lunch = st.selectbox("Lunch", options=["standard", "free/reduced"])
        test_preparation_course = st.selectbox("Test Preparation Course", options=["none", "completed"])
        reading_score = st.number_input("Reading Score", min_value=0, max_value=100, value=50)
        writing_score = st.number_input("Writing Score", min_value=0, max_value=100, value=50)

        submit_button = st.form_submit_button(label="Predict")

    if submit_button:
        # Prepare input data for prediction
        input_data = pd.DataFrame(
            [[gender, race_ethnicity, parental_level_of_education, lunch, test_preparation_course, reading_score, writing_score]],
            columns=["gender", "race_ethnicity", "parental_level_of_education", "lunch", "test_preparation_course", "reading_score", "writing_score"]
        )

        # Load the pipeline
        pipeline = load_pipeline('linear_regression_pipeline.pkl')

        # Make predictions on the user input
        predictions = predict(input_data, pipeline)

        # Display the prediction
        st.write(f"Predicted Math Score: {predictions[0]}")

def predict(input_data, pipeline):
    return pipeline.predict(input_data)

if __name__ == "__main__":
    main()
