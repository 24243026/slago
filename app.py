
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

st.set_page_config(page_title="Engagement Predictor", layout="wide")

st.title("📊 Social Media Engagement Prediction System")
st.write("Predict whether a post will receive **High Engagement** or **Low Engagement**")

@st.cache_data
def load_data():
    df = pd.read_csv(r"C:\Users\NODE1\PycharmProjects\pythonProject\social_media_engagement_dataset.csv")
    return df

df = load_data()

df.fillna(method='ffill', inplace=True)


df["Engagement_Score"] = df["Likes"] + df["Comments"] + df["Shares"]

median_value = df["Engagement_Score"].median()

df["Engagement_Label"] = df["Engagement_Score"].apply(
    lambda x: 1 if x >= median_value else 0
)

for col in df.select_dtypes(include=['object']).columns:
    df[col] = LabelEncoder().fit_transform(df[col])


X = df.drop(["Engagement_Label"], axis=1)
y = df["Engagement_Label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = RandomForestClassifier(n_estimators=150, random_state=42)
model.fit(X_train_scaled, y_train)

pred = model.predict(X_test_scaled)
acc = accuracy_score(y_test, pred)

menu = st.sidebar.selectbox(
    "Navigation",
    ["Dataset", "EDA Visualization", "Model Performance", "Prediction"]
)


if menu == "Dataset":

    st.subheader("📂 Dataset Preview")
    st.dataframe(df.head())

    st.write("Dataset Shape:", df.shape)

    st.subheader("📊 Dataset Summary")
    st.write(df.describe())

elif menu == "EDA Visualization":

    st.subheader("📊 Engagement Score Distribution")

    fig1, ax1 = plt.subplots()
    sns.histplot(df["Engagement_Score"], bins=30, kde=True, ax=ax1)
    st.pyplot(fig1)

    st.subheader("🔥 Correlation Heatmap")

    fig2, ax2 = plt.subplots(figsize=(10,6))
    sns.heatmap(df.corr(), annot=True, cmap="coolwarm", ax=ax2)
    st.pyplot(fig2)

elif menu == "Model Performance":

    st.subheader("🤖 Random Forest Model Performance")

    st.success(f"✅ Model Accuracy: {acc:.2f}")

    st.subheader("Confusion Matrix")
    st.write(confusion_matrix(y_test, pred))

    st.subheader("Classification Report")
    st.text(classification_report(y_test, pred))

elif menu == "Prediction":

    st.subheader("🔮 Predict Engagement for New Post")

    input_data = {}

    for col in X.columns:
        default_val = float(df[col].mean())
        input_data[col] = st.number_input(f"{col}", value=default_val)

    if st.button("Predict Engagement"):

        new_df = pd.DataFrame([input_data])

        new_scaled = scaler.transform(new_df)

        result = model.predict(new_scaled)[0]
        prob = model.predict_proba(new_scaled)

        st.subheader("Prediction Result")

        if result == 1:
            st.success("🔥 High Engagement Post")
        else:
            st.error("⚠️ Low Engagement Post")


        st.write(f"Confidence: {np.max(prob)*100:.2f}%")
