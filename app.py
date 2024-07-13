import streamlit as st
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import seaborn as sns

current_dir = os.path.dirname(__file__)
csv_path = os.path.join(current_dir, "../assets/wine.csv")


with open('../assets/wine_quality.pkl', 'rb') as file:
    model = pickle.load(file)
ds = pd.read_csv(csv_path)


def prediction(datavalues):
    result = model.predict(datavalues)
    return result


def info_graphics():
    st.write("### About Dataset")
    st.write("The Wine Quality dataset consists of data for two variants of Portuguese wines: red and white. The data was collected to analyze factors that influence wine quality and to develop models that can predict wine quality based on its chemical properties.")
    st.write("#### Fixed and Volatile acidity")
    st.write("fixed_acidity: Tartaric acid content, contributing to the total acidity of wine (range: 4.0 - 15.0 g/dm³).")
    st.write("volatile_acidity: Acetic acid content, which can give wine an unpleasant vinegar taste if too high (range: 0.1 - 1.5 g/dm³).")
    st.write("#### Critic acid")
    st.write("citric_acid: Citric acid content, adding freshness and flavor to wine (range: 0.0 - 1.0 g/dm³).")
    st.write("#### Residual sugar")
    st.write("residual_sugar: Residual sugar content, the amount of sugar remaining after fermentation (range: 0.6 - 65.8 g/dm³).")
    st.write("#### Free and Total Sulfer Dioxide")
    st.write("free_sulfur_dioxide: Free SO₂ content, used as a preservative to prevent spoilage and oxidation (range: 1 - 72 mg/dm³).")
    st.write("total_sulfur_dioxide: Total SO₂ content, including both free and bound forms (range: 6 - 289 mg/dm³).")
    st.write("#### Density")
    st.write("density: Density of the wine, related to alcohol and sugar content (range: 0.990 - 1.004 g/cm³).")
    st.write("#### pH")
    st.write("pH: pH value, indicating the acidity or basicity of the wine (range: 2.7 - 4.0).")
    st.write("#### Alcohol")
    st.write("alcohol: Alcohol content, expressed as a percentage by volume (range: 8.0% - 14.9%).")
    st.write("#### Sulphates and Chlorides")
    data = {
    'Attribute': ['sulphates', 'chlorides'],
    'Description': [
        "Potassium sulphate content, contributing to the wine's antimicrobial and antioxidant properties (range: 0.22 - 1.15 g/dm³).",
        "Sodium chloride content, which can affect the wine's taste and preservation (range: 0.009 - 0.346 g/dm³)."]}
    df = pd.DataFrame(data)
    st.table(df)
    st.write("## Pie Chart")
    wine_counts = ds['type'].value_counts()

    # Create a pie chart
    fig, ax = plt.subplots()
    ax.pie(wine_counts, labels=wine_counts.index, autopct='%1.1f%%',
           shadow=False, startangle=90)
    ax.axis('equal')
    st.pyplot(fig)

    st.write("## Countplot of Wine Quality")
    fig, ax = plt.subplots()
    sns.countplot(x='quality', data=ds, ax=ax)
    ax.set_title('Count of Wine Quality')
    ax.set_xlabel('Quality')
    ax.set_ylabel('Count')
    st.pyplot(fig)

    st.write("## Scatter Plot")

    # Create a scatter plot
    fig, ax = plt.subplots()
    sns.scatterplot(x='alcohol', y='residual sugar',
                    data=ds, ax=ax, hue='type')
    ax.set_title('Scatter Plot of Alcohol Content and Residual Sugar')
    ax.set_xlabel('Alcohol Content')
    ax.set_ylabel('Residual Sugar')

    st.pyplot(fig)


def input_tab():
    options = {"Red": 0, "White": 1}
    type = st.radio("Wine Type", list(options.keys()), horizontal=True)
    fixed_acidity = st.number_input(
        "Fixed acidity:", min_value=3.0, max_value=16.0, format="%.2f")
    volatile_acidity = st.number_input(
        "Volatile acidity:", min_value=0.0, max_value=2.0, format="%.2f")
    citric_acid = st.number_input(
        "Citric acid:", min_value=0.0, max_value=2.0, format="%.2f")
    residual_sugar = st.number_input(
        "Residual sugar:", min_value=0.0, max_value=68.0, format="%.2f")
    cl = st.number_input("Chlorides:", min_value=0.0,
                         max_value=1.0, format="%.2f")
    free_so2 = st.number_input(
        "Free sulfur dioxide:", min_value=1.0, max_value=295.0, format="%.2f")
    tot_so2 = st.number_input("Total sulfur dioxide:",
                              min_value=6.0, max_value=450.0, format="%.2f")
    den = st.number_input("Density:", min_value=0.5,
                          max_value=2.0, format="%.2f")
    ph = st.number_input("pH:", min_value=2.0, max_value=5.0, format="%.2f")
    sulphate = st.number_input(
        "Sulphates:", min_value=0.0, max_value=2.0, format="%.2f")
    alcohol = st.number_input(
        "Alcohol content:", min_value=8.0, max_value=16.0, format="%.2f")
    datavalues = [[options[type], fixed_acidity, volatile_acidity, citric_acid,
                   residual_sugar, cl, free_so2, tot_so2, den, ph, sulphate, alcohol]]

    predict = st.button('Predict')
    if predict and datavalues:
        result = prediction(datavalues)
        # st.write("Predicted Quality:",result[0])
        st.markdown(f"""
    <style>
    .result {{
        font-size: 50px;  /* Increase font size */
        text-align: center;  /* Center align text */
    }}
    </style>
    <div class="result">
        Predicted Quality: {result[0]}
    </div>
""", unsafe_allow_html=True)
   

def main():
    st.title("Wine Quality Prediction")
    tabs = ["Info Graphics", "Input"]
    selected_tab = st.sidebar.radio("Choose a tab", tabs)
    if selected_tab == "Info Graphics":
        info_graphics()
    elif selected_tab == "Input":
        input_tab()


if __name__ == "__main__":
    main()
