import streamlit as st
import pandas as pd
import pickle
import os

#Title
st.title("Midterm Project")

#Loading Model
def load_model():
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
    return model

model = load_model()

#Note: I didn't come up with this code. But I've tried to understand.
def get_options():
    preprocesser = model.named_steps['preprocesser']
    #Taking preprocessor column_transformer, from full_pipeline. 
    cat_transformer = preprocesser.named_transformers_['cat']
    #Taking categorical transformer from preprocessor.
    onehot = cat_transformer.named_steps['onehot']
    #Taking onehot encoding step from categorical transformer pipeline. 
    cat_options = onehot.categories_
    #Extracting categories. Similar to .unique().
    
    #Doing similar step for education column. 
    edu_transformer = preprocesser.named_transformers_['education']
    ordinal = edu_transformer.named_steps['ordinal']
    edu_options = ordinal.categories_[0]

    options = {
        'workclass': list(cat_options[0]),
        'marital-status': list(cat_options[1]),
        'occupation': list(cat_options[2]),
        'relationship': list(cat_options[3]),
        'race': list(cat_options[4]),
        'sex': list(cat_options[5]),
        'native-country': list(cat_options[6]),
        'education': list(edu_options)
        }
    return options  

options = get_options()


#This code as well, I've taken from LLM. I can understand it but, I didn't come up with it. 
#The 4 functions below are for numeric features. 
def get_age_range():
    preprocessor = model.named_steps['preprocessor']
    num_transformer = preprocessor.named_transformers_['num']
    scaler = num_transformer.named_steps['scaler']
    mean_value = scaler.mean_[0]
    std_value = scaler.scale_[0]
    min_value = int(max(17, mean_value - (3 * std_value)))
    max_value = int(min(100, mean_value - (3 * std_value)))
    default_value = int(mean_value)
    return min_value, max_value, default_value

min_age, max_age, default_age = get_age_range()

def get_capital_gain_range():
    preprocessor = model.named_steps['preprocessor']
    num_transformer = preprocessor.named_transformers_['num']
    scaler = num_transformer.named_steps['scaler']
    mean_value = scaler.mean_[1]
    std_value = scaler.scale_[1]
    min_value = 0
    max_value = int(min(200000, mean_value - (3 * std_value)))
    default_value = 0
    return min_value, max_value, default_value

min_gain, max_gain, default_gain = get_capital_gain_range()

def get_capital_loss_range():
    preprocessor = model.named_steps['preprocessor']
    num_transformer = preprocessor.named_transformers_['num']
    scaler = num_transformer.named_steps['scaler']
    mean_value = scaler.mean_[2]
    std_value = scaler.scale_[2]
    min_value = 0
    max_value = int(min(10000, mean_value - (3 * std_value)))
    default_value = 0
    return min_value, max_value, default_value

min_loss, max_loss, default_loss = get_capital_loss_range()

def get_hours_range():
    preprocessor = model.named_steps['preprocessor']
    num_transformer = preprocessor.named_transformers_['num']
    scaler = num_transformer.named_steps['scaler']
    mean_value = scaler.mean_[3]
    std_value = scaler.scale_[3]
    min_value = int(max(1, mean_value - (3 * std_value)))
    max_value = int(min(100, mean_value - (3 * std_value)))
    default_value = int(mean_value)
    return min_value, max_value, default_value

min_hours, max_hours, default_hours = get_hours_range()

#Side Bar
logo_path = "Images/pulogo.jpg"
if os.path.exist(logo_path):
    st.sidebar.image(logo_path, width = 150)
st.sidebar.markdown("**Name: Tar Rein Thway**")
st.sidebar.markdown("**ID: PIUS20220022**")

#Inputs
st.header("Please Enter Information")
age = st.number_input("Age", min_age, max_age, default_age)

workclass = st.selectbox("Work Class", options['workclass'] )

education = st.selectbox("Education", options['education'])

marital_status = st.selectbox("Marital Status", options['marital-status'])

occupation = st.selectbox("Occupation", options['occupation'])

relationship = st.selectbox("Relationship", options['relationship'])

race = st.selectbox("Race", options['race'])

sex = st.selectbox("Sex", options['sex'])

capital_gain = st.number_input("Capital Gain", min_gain, max_gain, default_gain)

capital_loss = st.number_input("Capital Loss", min_loss, max_loss, default_loss)

hours_per_week = st.number_input("Hours per Week", min_hours, max_hours, default_hours)

native_country = st.selectbox("Native Country", options['native-country'])



#Predict
if st.button("Predict"):
    input_data = pd.DataFrame({
        'age': [age],
        'workclass': [workclass],
        'education': [education],
        'marital-status': [marital_status],
        'occupation': [occupation],
        'relationship': [relationship],
        'race': [race],
        'sex': [sex],
        'capital-gain': [capital_gain],
        'capital-loss': [capital_loss],
        'hours-per-week': [hours_per_week],
        'native-country': [native_country]
    })

    result = model.predict(input_data)[0]

    if result == 1:
        st.success(f"Oh ma godd, this person makes more than 50 thousands a year! RICHHH!!💰💰💵💵🤑🤑")
    else:
        st.info(f"Even if it's not 50 thousands, its still good money tho🤨🤭")
