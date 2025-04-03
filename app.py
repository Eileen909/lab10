import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, silhouette_score
import os

# Set page config
st.set_page_config(page_title="Women's Clothing E-Commerce Analysis", layout="wide")

# Title
st.title("Women's Clothing E-Commerce Analysis Dashboard")

# Sidebar
st.sidebar.header("Navigation")
page = st.sidebar.radio("Go to", ["Data Overview", "Data Cleaning", "EDA", "Machine Learning"])

# Function to generate synthetic data
def generate_data():
    np.random.seed(42)
    n_samples = 23486
    
    data = {
        'Age': np.random.randint(18, 70, n_samples),
        'Title': [f'Review {i}' for i in range(n_samples)],
        'Review_Text': [f'This is review text {i}' for i in range(n_samples)],
        'Rating': np.random.randint(1, 6, n_samples),
        'Recommended_IND': np.random.randint(0, 2, n_samples),
        'Positive_Feedback_Count': np.random.randint(0, 100, n_samples),
        'Division_Name': np.random.choice(['General', 'General Petite', 'Initmates'], n_samples),
        'Department_Name': np.random.choice(['Tops', 'Dresses', 'Bottoms', 'Intimate', 'Jackets'], n_samples),
        'Class_Name': np.random.choice(['Dresses', 'Knits', 'Blouses', 'Sweaters', 'Jeans'], n_samples)
    }
    
    return pd.DataFrame(data)

# Load or generate data
@st.cache_data
def load_data():
    try:
        # Try to load from the data directory first
        data_path = os.path.join('data', 'Womens_Clothing_E-Commerce_Reviews.csv')
        if os.path.exists(data_path):
            df = pd.read_csv(data_path)
        else:
            # Try to load from the root directory
            df = pd.read_csv('Womens_Clothing_E-Commerce_Reviews.csv')
    except:
        df = generate_data()
        # Create data directory if it doesn't exist
        os.makedirs('data', exist_ok=True)
        df.to_csv(os.path.join('data', 'Womens_Clothing_E-Commerce_Reviews.csv'), index=False)
    return df

# Load data
df = load_data()

if page == "Data Overview":
    st.header("Dataset Overview")
    st.write("### Sample Data")
    st.dataframe(df.head())
    
    st.write("### Dataset Information")
    st.write(f"Number of rows: {df.shape[0]}")
    st.write(f"Number of columns: {df.shape[1]}")
    
    st.write("### Column Descriptions")
    st.write("""
    - Age: Reviewer's age
    - Title: Review title
    - Review_Text: Review content
    - Rating: Product rating (1-5)
    - Recommended_IND: Whether product is recommended (0/1)
    - Positive_Feedback_Count: Number of positive feedbacks
    - Division_Name: Product division
    - Department_Name: Product department
    - Class_Name: Product class
    """)

elif page == "Data Cleaning":
    st.header("Data Cleaning and Wrangling")
    
    # Missing values analysis
    st.subheader("Missing Values Analysis")
    missing_values = df.isnull().sum()
    st.dataframe(missing_values)
    
    # Data cleaning options
    st.subheader("Data Cleaning Options")
    cleaning_option = st.selectbox("Select cleaning operation", 
                                 ["Remove missing values", "Fill missing values with mean", "Fill missing values with mode"])
    
    if st.button("Apply Cleaning"):
        if cleaning_option == "Remove missing values":
            cleaned_df = df.dropna()
        elif cleaning_option == "Fill missing values with mean":
            cleaned_df = df.fillna(df.mean())
        else:
            cleaned_df = df.fillna(df.mode().iloc[0])
        
        st.write("Cleaned Data Sample")
        st.dataframe(cleaned_df.head())

elif page == "EDA":
    st.header("Exploratory Data Analysis")
    
    # Age distribution
    st.subheader("Age Distribution")
    fig_age = px.histogram(df, x='Age', nbins=30, title='Age Distribution')
    st.plotly_chart(fig_age)
    
    # Rating distribution
    st.subheader("Rating Distribution")
    fig_rating = px.pie(df, names='Rating', title='Rating Distribution')
    st.plotly_chart(fig_rating)
    
    # Department distribution
    st.subheader("Department Distribution")
    fig_dept = px.bar(df['Department_Name'].value_counts(), title='Department Distribution')
    st.plotly_chart(fig_dept)
    
    # Correlation analysis
    st.subheader("Correlation Analysis")
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    corr_matrix = df[numeric_cols].corr()
    fig_corr = px.imshow(corr_matrix, title='Correlation Matrix')
    st.plotly_chart(fig_corr)

elif page == "Machine Learning":
    st.header("Machine Learning Analysis")
    
    ml_task = st.selectbox("Select ML Task", ["Regression", "Clustering", "Classification"])
    
    if ml_task == "Regression":
        st.subheader("Rating Prediction")
        X = df[['Age', 'Positive_Feedback_Count']]
        y = df['Rating']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        model = LinearRegression()
        model.fit(X_train, y_train)
        score = model.score(X_test, y_test)
        
        st.write(f"Model R2 Score: {score:.4f}")
        
    elif ml_task == "Clustering":
        st.subheader("Customer Segmentation")
        X = df[['Age', 'Rating', 'Positive_Feedback_Count']]
        n_clusters = st.slider("Number of clusters", 2, 5, 3)
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(X)
        
        fig_clusters = px.scatter_3d(df, x='Age', y='Rating', z='Positive_Feedback_Count',
                                   color=clusters, title='Customer Clusters')
        st.plotly_chart(fig_clusters)
        
    else:  # Classification
        st.subheader("Recommendation Prediction")
        X = df[['Age', 'Rating', 'Positive_Feedback_Count']]
        y = df['Recommended_IND']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        model = RandomForestClassifier(random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        st.write(f"Model Accuracy: {accuracy:.4f}")

# Footer
st.markdown("---")
st.markdown("Created with ❤️ using Streamlit") 