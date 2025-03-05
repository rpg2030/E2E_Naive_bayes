import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Set Streamlit page config
st.set_page_config(
    page_title="Diabetes Predictor",
    page_icon="💉",
    layout="wide"
)

# Custom CSS for styling
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .stButton>button {
        background-color: #ff4b4b;
        color: white;
        font-size: 18px;
        border-radius: 10px;
        padding: 10px 20px;
    }
    .stButton>button:hover {
        background-color: #ff0000;
    }
    </style>
""", unsafe_allow_html=True)

# Title with styling
st.title("💉 Diabetes Prediction using Naive Bayes")

# Load the dataset
@st.cache_data
def load_data():
    return pd.read_csv('Diabetespred.csv')

df = load_data()

# Display dataset
st.subheader("📊 Dataset Overview")
st.dataframe(df.style.background_gradient(cmap="Blues"))  # Beautify table

# Split dataset into features and target
X = df.drop('Outcome', axis=1)
y = df['Outcome']

# Apply SMOTE to balance dataset
smote = SMOTE()
X, y = smote.fit_resample(X, y)

# Feature scaling
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Train the Naive Bayes model
@st.cache_resource
def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y,random_state=42)
    model = GaussianNB()
    model.fit(X_train, y_train)
    return model, X_test, y_test

model, X_test, y_test = train_model(X, y)

# Model accuracy
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# Display accuracy using metric
st.subheader("📈 Model Performance")
st.metric(label="Model Accuracy", value=f"{accuracy:.2%}")

# Sidebar for user input
st.sidebar.title("🔍 Enter Patient Details")

pregnancies = st.sidebar.number_input("Pregnancies", min_value=0, max_value=20, value=1)
glucose = st.sidebar.number_input("Glucose", min_value=0, max_value=200, value=100)
blood_pressure = st.sidebar.number_input("Blood Pressure", min_value=0, max_value=122, value=70)
skin_thickness = st.sidebar.number_input("Skin Thickness", min_value=0, max_value=99, value=20)
insulin = st.sidebar.number_input("Insulin", min_value=0, max_value=846, value=80)
bmi = st.sidebar.number_input("BMI", min_value=0.0, max_value=67.1, value=25.0)
diabetes_pedigree = st.sidebar.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=2.5, value=0.5)
age = st.sidebar.number_input("Age", min_value=0, max_value=120, value=30)

# Prediction button
if st.sidebar.button("🔮 Predict"):
    input_data = pd.DataFrame([{
        'Pregnancies': pregnancies, 'Glucose': glucose, 'BloodPressure': blood_pressure,
        'SkinThickness': skin_thickness, 'Insulin': insulin, 'BMI': bmi,
        'DiabetesPedigreeFunction': diabetes_pedigree, 'Age': age
    }])
    
    # Make prediction
    prediction = model.predict(input_data)
    if prediction[0] == 1:
        st.error("🚨 The model predicts that the person **HAS** diabetes.")
    else:
        st.success("✅ The model predicts that the person **DOES NOT HAVE** diabetes.")

# Feature visualization
st.subheader("📊 Data Visualizations")

# KDE Plot (Density Plot)
st.subheader("🔹 Distribution of Features (KDE Plot)")
feature = st.selectbox("Select a feature to visualize:", df.columns[:-1])

fig, ax = plt.subplots(figsize=(10, 6))
sns.kdeplot(data=df, x=feature, hue='Outcome', fill=True, palette=["#ff4b4b", "#4caf50"], ax=ax)
plt.title(f"Density Plot of {feature} by Diabetes Outcome")
plt.xlabel(feature)
plt.ylabel("Density")
st.pyplot(fig)

# Correlation Heatmap
st.subheader("🔹 Correlation Heatmap")
fig, ax = plt.subplots(figsize=(12, 8))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5, ax=ax)
plt.title("Feature Correlation Heatmap")
st.pyplot(fig)

# Pairplot
st.subheader("🔹 Pairplot of Features")
st.write("This plot helps visualize relationships between different features.")
fig = sns.pairplot(df, hue="Outcome", diag_kind="kde", palette=["#ff4b4b", "#4caf50"])
st.pyplot(fig)

# Count Plot (Diabetes vs. Non-Diabetes Cases)
st.subheader("🔹 Count of Diabetes vs. Non-Diabetes Cases")
fig, ax = plt.subplots(figsize=(8, 5))
sns.countplot(data=df, x="Outcome", palette=["#4caf50", "#ff4b4b"])
plt.xticks([0, 1], ['Non-Diabetic', 'Diabetic'])
plt.ylabel("Count")
plt.title("Distribution of Diabetes Cases")
st.pyplot(fig)

# Box Plot (To Detect Outliers)
st.subheader("🔹 Box Plot for Outlier Detection")
feature_box = st.selectbox("Select a feature for Box Plot:", df.columns[:-1])
fig, ax = plt.subplots(figsize=(8, 5))
sns.boxplot(data=df, x="Outcome", y=feature_box, palette=["#4caf50", "#ff4b4b"])
plt.title(f"Box Plot of {feature_box} by Diabetes Outcome")
st.pyplot(fig)


