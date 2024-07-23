import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st 

def plot_pie_chart(ds):
    st.write("## Pie Chart")
    wine_counts = ds['type'].value_counts()

    # Create a pie chart
    fig, ax = plt.subplots()
    ax.pie(wine_counts, labels=wine_counts.index, autopct='%1.1f%%',
           shadow=False, startangle=90)
    ax.axis('equal')
    st.pyplot(fig)


def plot_countplot(ds):
    st.write("## Countplot of Wine Quality")
    fig, ax = plt.subplots()
    sns.countplot(x='quality', data=ds, ax=ax)
    ax.set_title('Count of Wine Quality')
    ax.set_xlabel('Quality')
    ax.set_ylabel('Count')
    st.pyplot(fig)
    st.write("The dataset shows a predominance of medium quality ratings (5 and 6), with fewer instances of both low (3 or 4) and high quality (7 or higher) wines.")


def plot_scatterplot(ds, wine_type):
    filtered_data = ds[ds['type'] == wine_type]
    fig, ax = plt.subplots()
    sns.scatterplot(x='alcohol', y='residual sugar', data=filtered_data, ax=ax)
    ax.set_title(f'Scatter Plot of Alcohol Content and Residual Sugar for {wine_type.capitalize()} Wine')
    ax.set_xlabel('Alcohol Content')
    ax.set_ylabel('Residual Sugar')
    st.pyplot(fig)
    st.write("The scatter plots reveal that the relationship between alcohol content and residual sugar differs significantly between white and red wines. White wine shows a clear inverse relationship between alcohol content and residual sugar, indicating that as the alcohol content increases, the residual sugar tends to decrease, while red wine does not show a strong correlation between alcohol content and residual sugar, suggesting that these two variables vary independently.")

def plot_boxplot(ds):
    st.write("## Box Plot of Quality by Wine Type")
    fig, ax = plt.subplots()
    sns.boxplot(x='type', y='quality', data=ds, ax=ax)
    ax.set_title('Box Plot of Wine Quality by Type')
    ax.set_xlabel('Wine Type')
    ax.set_ylabel('Quality')
    st.pyplot(fig)
    st.write("Both wine types exhibit outliers, suggesting some wines have quality scores significantly different from the majority. The distribution of quality scores is quite similar between white and red wines, with no notable differences in central tendencies or variability.")


def vol_acidity(ds):
    st.write("## Volatile Acidity vs Wine Quality")
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(x='quality', y='volatile acidity', data=ds, ax=ax)
    ax.set_title('Volatile Acidity vs Quality')
    ax.set_xlabel('Wine Quality')
    ax.set_ylabel('Volatile Acidity')
    st.pyplot(fig)
    st.write(" Volatile acidity and wine quality are inversely proportional. If volatile acidity is high, the quality is typically low.")
    
def citric_acid(ds):
    st.write("## Citric Acid vs Wine Quality")
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(x='quality', y='citric acid', data=ds, ax=ax)
    ax.set_title('Citric Acid vs Quality')
    ax.set_xlabel('Wine Quality')
    ax.set_ylabel('Citric Acid')
    st.pyplot(fig)
    st.write("Citric acid and wine quality are directly proportional. If citric acid content is higher, the quality of wine is typically higher.")

def box_plot(ds):
    st.write("## Box Plot of Alcohol Content by Wine Quality")
    plt.figure(figsize=(12, 8))
    sns.boxplot(x='quality', y='alcohol', data=ds, palette='muted')
    plt.title('Box Plot of Alcohol Content by Wine Quality')
    plt.xlabel('Wine Quality')
    plt.ylabel('Alcohol Content')
    st.pyplot(plt)
    st.write("The plot shows a consistent increase in alcohol content with higher wine quality ratings.")