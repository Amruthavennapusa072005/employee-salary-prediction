import gradio as gr

# Load everything
model = joblib.load("income_model.pkl")
encoders = joblib.load("encoders.pkl")
target_encoder = joblib.load("target_encoder.pkl")

def predict_income(age, workclass, fnlwgt, education, educational_num, marital_status,
                   occupation, relationship, race, gender, capital_gain,
                   capital_loss, hours_per_week, native_country):

    input_data = {
        'age': [age],
        'workclass': [encoders['workclass'].transform([workclass])[0]],
        'fnlwgt': [fnlwgt],
        'education': [encoders['education'].transform([education])[0]],
        'educational-num': [educational_num],
        'marital-status': [encoders['marital-status'].transform([marital_status])[0]],
        'occupation': [encoders['occupation'].transform([occupation])[0]],
        'relationship': [encoders['relationship'].transform([relationship])[0]],
        'race': [encoders['race'].transform([race])[0]],
        'gender': [encoders['gender'].transform([gender])[0]],
        'capital-gain': [capital_gain],
        'capital-loss': [capital_loss],
        'hours-per-week': [hours_per_week],
        'native-country': [encoders['native-country'].transform([native_country])[0]]
    }

    df_input = pd.DataFrame(input_data)
    prediction = model.predict(df_input)[0]
    return target_encoder.inverse_transform([prediction])[0]
input_components = [
    gr.Slider(17, 90, label="Age"),
    gr.Dropdown(encoders['workclass'].classes_.tolist(), label="Workclass"),
    gr.Number(label="Fnlwgt"),
    gr.Dropdown(encoders['education'].classes_.tolist(), label="Education"),
    gr.Slider(1, 16, label="Educational Number"),
    gr.Dropdown(encoders['marital-status'].classes_.tolist(), label="Marital Status"),
    gr.Dropdown(encoders['occupation'].classes_.tolist(), label="Occupation"),
    gr.Dropdown(encoders['relationship'].classes_.tolist(), label="Relationship"),
    gr.Dropdown(encoders['race'].classes_.tolist(), label="Race"),
    gr.Dropdown(encoders['gender'].classes_.tolist(), label="Gender"),
    gr.Number(label="Capital Gain"),
    gr.Number(label="Capital Loss"),
    gr.Slider(1, 99, label="Hours per Week"),
    gr.Dropdown(encoders['native-country'].classes_.tolist(), label="Native Country"),
]

gr.Interface(
    fn=predict_income,
    inputs=input_components,
    outputs=gr.Text(label="Predicted Income"),
    title="Employee Income Prediction App"
).launch(share=True)
