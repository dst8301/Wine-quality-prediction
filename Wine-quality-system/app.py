import streamlit as st
import pickle
import numpy as np
import pandas as pd
import os
import seaborn as sns
from plot import plot_pie_chart, plot_countplot, plot_scatterplot, plot_boxplot,vol_acidity,citric_acid,box_plot
current_dir = os.path.dirname(__file__)
csv_path = os.path.join(current_dir, "../assets/wine.csv")


with open('../assets/wine_quality.pkl', 'rb') as file:
    model = pickle.load(file)
ds = pd.read_csv(csv_path)


def prediction(datavalues):
    result = model.predict(datavalues)
    return result

def app():
    image_path = "wine.jpg"
    st.image(image_path, use_column_width=True)
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.write("A Wine Quality Prediction app is a tool designed to assess the quality of wine based on its chemical properties. Utilizing advanced machine learning algorithms, the app predicts the quality score of a wine sample on a scale typically ranging from 3 to 9. The app offers an intuitive interface where users can input the chemical attributes of the wine. Wine quality is influenced by a variety of chemical factors, including acidity, pH, alcohol content, and other attributes. Predicting wine quality traditionally involves expert sensory evaluation, which can be subjective and inconsistent. The Wine Quality Prediction app provides an objective, data-driven approach to quality assessment. Beyond providing a quality score, the app helps users understand the influence of various chemical factors on wine quality. By analyzing these attributes, the app offers actionable insights that can be used to enhance wine production processes and ensure consistency in the final product.")
    st.write("The Wine Quality Prediction app evaluates the quality of a wine sample based on a set of critical chemical attributes. They are:")
    st.markdown("""
    - Fixed acidity
    - Volatile acidity
    - Citric acid
    - Residual sugar
    - Chlorides
    - Free sulfur dioxide
    - Total sulfur dioxide
    - Density
    - pH
    - Sulphates
    - Alcohol
    """)
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
    
    
    plot_pie_chart(ds)
    plot_countplot(ds)
    st.write("## Scatter Plot")
    wine_type_options = {"Red": "red", "White": "white"}
    selected_wine_type = st.radio("#### Select Wine Type for Scatter Plot", list(wine_type_options.keys()),horizontal=True)
    plot_scatterplot(ds, wine_type_options[selected_wine_type])
    plot_boxplot(ds)
    vol_acidity(ds)
    citric_acid(ds)
    box_plot(ds)

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

# Dictionary mapping quality scores to descriptions
    # quality_descriptions = {
    # 3: "Very poor quality wine.",
    # 4: "Poor quality wine.",
    # 5: "Below average quality wine ",
    # 6: "Average quality wine.",
    # 7: "Good quality wine.",
    # 8: "Very good quality wine.",
    # 9: "Excellent quality wine."
    # }

    # predict = st.button('Predict')
    # if predict and datavalues:
    #     result = prediction(datavalues)
    #     predicted_quality = result[0]
    #     description = quality_descriptions.get(predicted_quality, "Unknown quality.")

    #     st.markdown(f"""
    #     <style>
    #     .result {{
    #         font-size: 40px;  /* Increase font size */
    #         text-align: center;  /* Center align text */
    #     }}
    #     .description {{
    #         font-size: 20px;  /* Smaller font size for description */
    #         text-align: center;  /* Center align text */
    #         margin-top: 20px;  /* Add some space above the description */
    #     }}
    #     </style>
    #     <div class="result">
    #         Predicted Quality: {predicted_quality}
    #     </div>
    #     <div class="description">
    #         {description}
    #     </div>
    #     """, unsafe_allow_html=True)
    

# Detailed descriptions for each wine quality rating based on chemical properties
    quality_descriptions = {
    3: ("Very Poor Quality Wine", "Typically very low alcohol content, high residual sugar, and excessive volatile acidity, leading to an imbalanced and unstable profile."),
    4: ("Poor Quality Wine", "Low alcohol content and noticeable imbalances in residual sugar and acidity, with slight spoilage notes."),
    5: ("Below Average Quality Wine", "Slightly below average alcohol content with minor imbalances in residual sugar and acidity, indicating some flaws but drinkable characteristics."),
    6: ("Average Quality Wine", "Balanced chemical properties with average alcohol content, well-matched residual sugar and acidity, and normal density, indicating a decent fermentation process."),
    7: ("Good Quality Wine", "Higher alcohol content, well-balanced residual sugar and acidity, low volatile acidity, and optimal pH levels, reflecting good preservation and fermentation practices."),
    8: ("Very Good Quality Wine", "High alcohol content, perfectly balanced residual sugar and acidity, very low volatile acidity, and ideal pH levels, indicating excellent preservation and complexity."),
    9: ("Excellent Quality Wine", "Very high alcohol content, exceptionally well-balanced residual sugar and acidity, extremely low volatile acidity, and perfect pH levels, reflecting superior preservation and flawless fermentation practices, resulting in the highest purity and quality.")
}

    predict = st.button('Predict')
    if predict and datavalues:
        result = prediction(datavalues)
        predicted_quality = result[0]
        heading, description = quality_descriptions.get(predicted_quality, ("Unknown Quality", "No description available."))

        st.markdown(f"""
        <style>
        .result {{
            font-size: 40px;  /* Increase font size */
            text-align: center;  /* Center align text */
        }}
        .heading {{
            font-size: 30px;  /* Font size for heading */
            text-align: center;  /* Center align text */
            margin-top: 20px;  /* Add some space above the heading */
        }}
        .description {{
            font-size: 20px;  /* Smaller font size for description */
            text-align: left;  
            margin-top: 10px;  /* Add some space above the description */
        }}
        </style>
        <div class="result">
            Predicted Quality: {predicted_quality}
        </div>
        <div class="heading">
            {heading}
        </div>
        <div class="description">
            {description}
        </div>
        """, unsafe_allow_html=True)


def main():
    st.title("Wine Quality Prediction App")
    tabs = ["App Description","Info Graphics", "Input"]
    selected_tab = st.sidebar.radio("Choose a tab", tabs)
    if selected_tab == "App Description":
        app()
    elif selected_tab == "Info Graphics":
        info_graphics()
    elif selected_tab == "Input":
        input_tab()


if __name__ == "__main__":
    main()
