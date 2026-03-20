import pandas as pd
import numpy as np

from google.colab import drive
drive.mount('/content/drive')


file_path = "/content/drive/MyDrive/Delhi_Cleaned.csv"
df = pd.read_csv(file_path)

print("Dataset Loaded Successfully ✅\n")


print("🔹 BEFORE CLEANING:\n")
df.info()

print("\nMissing Values Before Cleaning:\n")
print(df.isnull().sum())


# Remove duplicate rows
df.drop_duplicates(inplace=True)

# Fill numeric missing values with mean
numeric_cols = df.select_dtypes(include=np.number).columns
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())

# Fill object (text) missing values with mode
object_cols = df.select_dtypes(include='object').columns
for col in object_cols:
    df[col] = df[col].fillna(df[col].mode()[0])


print("\n🔹 AFTER CLEANING:\n")
df.info()

print("\nMissing Values After Cleaning:\n")
print(df.isnull().sum())


cleaned_path = "/content/drive/MyDrive/Delhi_Cleaned_Final.csv"
df.to_csv(cleaned_path, index=False)

print("\nCleaned dataset saved successfully as Delhi_Cleaned_Final.csv ✅")

from google.colab import drive
import pandas as pd

# Mount Drive
drive.mount('/content/drive')

# Load cleaned dataset
df = pd.read_csv("/content/drive/MyDrive/Delhi_Cleaned.csv")

print("Dataset loaded successfully")

# Show first 5 rows
display(df.head())

# Show column names
print("\nColumns:")
print(df.columns)

# Drop date column (not needed for correlation)
df_numeric = df.drop(columns=["date"])

# Trend analysis
print("\nStatistics:")
display(df_numeric.describe())

# Correlation with aqi (lowercase!)
print("\nCorrelation with AQI:")
display(df_numeric.corr()["aqi"].sort_values(ascending=False))

print("\n✅ Module 2 completed successfully")

from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_absolute_error

# Select important features
X = df_numeric[['pm10','pm2.5','co','no2']]
y = df_numeric['aqi']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Model
model = XGBRegressor()
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Accuracy
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print("R2 Accuracy:", r2)
print("MAE:", mae)
r2_percent = r2 * 100
print("\nFinal Accuracy (%):", round(r2_percent, 2))


print("\n✅ Module 3 completed (Model trained & tested)")


import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBRegressor

st.set_page_config(page_title="Air Quality Dashboard", layout="wide")

# ---------------- DARK STYLE ----------------
st.markdown("""
<style>
.stApp {
    background-color: #0E1117;
}
section[data-testid="stSidebar"] {
    background-color: #161A22;
}
h1, h2, h3 {
    color: #4FC3F7;
}
p, label, div {
    color: #E0E0E0;
}
button {
    background-color: #4FC3F7 !important;
    color: black !important;
}
div[data-baseweb="input"] > div {
    background-color: #1E222B !important;
    color: white !important;
}
</style>
""", unsafe_allow_html=True)

st.title("🌍 Air Quality Monitoring Dashboard")

# ---------------- SIDEBAR ----------------

st.sidebar.header("Controls")

uploaded_file = st.sidebar.file_uploader("Upload Dataset", type=["csv"])

pollutant = st.sidebar.selectbox(
    "Select Pollutant",
    ["pm10","pm2.5","co","no2"]
)

threshold = st.sidebar.slider("Alert Threshold (AQI)",0,500,100)

admin_mode = st.sidebar.toggle("Admin Mode")

# ---------------- MAIN ----------------

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)
    df.columns = df.columns.str.strip().str.lower()

    st.subheader("Dataset Preview")

    # 🔥 DARK DATAFRAME
    st.dataframe(
        df.head().style.set_properties(
            **{
                'background-color': '#1E222B',
                'color': 'white',
                'border-color': '#444'
            }
        )
    )

    # KPI
    if 'aqi' in df.columns:
        col1,col2,col3 = st.columns(3)
        col1.metric("Avg AQI", int(df['aqi'].mean()))
        col2.metric("Max AQI", int(df['aqi'].max()))
        col3.metric("Min AQI", int(df['aqi'].min()))

    # 🔥 DARK CHART
    st.subheader("Pollution Trend")

    fig, ax = plt.subplots()
    fig.patch.set_facecolor('#0E1117')
    ax.set_facecolor('#1E222B')

    ax.plot(df[pollutant], color='#4FC3F7')

    ax.tick_params(colors='white')
    ax.spines['bottom'].set_color('white')
    ax.spines['left'].set_color('white')

    ax.set_title(pollutant.upper(), color='white')

    st.pyplot(fig)

    # ALERT
    if 'aqi' in df.columns:
        if len(df[df['aqi'] > threshold]) > 0:
            st.error("🚨 High AQI detected")
        else:
            st.success("✅ AQI within safe limit")

    # MODEL
    if all(col in df.columns for col in ['pm10','pm2.5','co','no2','aqi']):

        X = df[['pm10','pm2.5','co','no2']]
        y = df['aqi']

        model = XGBRegressor()
        model.fit(X,y)

        st.subheader("AQI Prediction")

        col1,col2 = st.columns(2)

        with col1:
            pm10 = st.number_input("PM10",0.0,500.0,100.0)
            pm25 = st.number_input("PM2.5",0.0,500.0,80.0)

        with col2:
            co = st.number_input("CO",0.0,50.0,1.0)
            no2 = st.number_input("NO2",0.0,200.0,40.0)

        if st.button("Predict AQI"):

            input_data = np.array([[pm10,pm25,co,no2]])
            prediction = model.predict(input_data)[0]

            st.subheader(f"Predicted AQI: {round(prediction,2)}")

            if prediction <= 50:
                st.success("Good")
            elif prediction <= 100:
                st.info("Moderate")
            elif prediction <= 200:
                st.warning("Poor")
            elif prediction <= 300:
                st.error("Very Poor")
            else:
                st.error("Severe")

    # ADMIN PANEL
    if admin_mode:

        st.subheader("Admin Dashboard")

        new_file = st.file_uploader(
            "Upload New Dataset",
            type=["csv"],
            key="admin_upload"
        )

        if new_file is not None:

            new_df = pd.read_csv(new_file)
            new_df.columns = new_df.columns.str.strip().str.lower()

            if all(col in new_df.columns for col in ['pm10','pm2.5','co','no2','aqi']):

                X_new = new_df[['pm10','pm2.5','co','no2']]
                y_new = new_df['aqi']

                model.fit(X_new,y_new)

                st.success("Model retrained successfully")

else:
    st.info("Upload dataset from sidebar to start")
