# ⚖️ PROJECT TITLE
# District-wise Crime Analytics & Prediction Dashboard – Odisha (2015)

# 🎯 OBJECTIVE (Layman Definition)
# To analyze the pattern of IPC crimes reported across districts of Odisha in 2015, visualize them interactively, and use machine learning to predict the Total Cognizable Crimes based on selected features (like Murder, Theft, Dowry Deaths, etc.).
# This helps understand which crime types affect total reported crimes the most — giving insights useful for data-driven governance, law students, or data science learners.


import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

st.set_page_config(page_title="Odisha IPC Crime Dashboard", layout="wide")
st.title("⚖️ Odisha Crime Analytics & Prediction Dashboard (2015)")
st.write("Explore, visualize, and predict IPC crime trends across Odisha districts.")

# -------------------------------
# STEP 1: Load Dataset
# -------------------------------
uploaded_file = st.file_uploader(
    r"dataset\District-Wise-IPC-Cases-Reported_odisha-2015_0.csv", type=["csv"]
)

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    # -------------------------------
    # STEP 2: Data Cleaning
    # -------------------------------
    st.subheader("🧹 Data Cleaning Summary")

    df = df.drop_duplicates()
    df.fillna(df.mean(numeric_only=True), inplace=True)
    st.write("Preview of Data: (first 5 rows)")
    st.dataframe(df.head())
    
    # -------------------------------
    # STEP 3: User Selections
    # -------------------------------
    st.subheader("📊 Visualization")

    # Let users choose multiple districts for comparison
    selected_districts = st.multiselect(
        "🏙️ Choose One or More Districts to Visualize",
        sorted(df["District"].unique()),
        default=["DCP BBSR", "Cuttack", "Puri", "Rourkela", "Sambalpur"],
    )

    columns = st.multiselect(
        "📊 Choose Columns for Visualization or ML",
        [col for col in df.columns if col not in ["District", "Total Cognizable IPC Crimes"]],
        default=["Murder", "Theft"],
    )

    st.write("📈 Bar Plot Visualization")
    # -------------------------------
    # STEP 4: Visualization
    # -------------------------------
    # Remove 'Total District(s)' from visualization
    viz_df = df[~df["District"].str.lower().str.contains("total")]

    # Filter only selected districts
    if selected_districts:
        viz_df = viz_df[viz_df["District"].isin(selected_districts)]

    st.bar_chart(viz_df.set_index("District")[columns])

    # -------------------------------
    # STEP 5: Machine Learning Prediction
    # -------------------------------
    st.subheader("🤖 Predict Total Cognizable IPC Crimes")

    model_choice = st.selectbox(
        "Choose Model", ["Linear Regression", "Decision Tree", "Random Forest"]
    )

    features = st.multiselect(
        "Select Columns for Model",
        [
            col
            for col in df.columns
            if col not in ["District", "Year", "Total Cognizable IPC Crimes"]
        ],
        default=["Murder", "Theft", "Dowry Deaths"],
    )

    # -----------------------------------
    # STEP 3: USER INPUTS
    # -----------------------------------
    # Year input (user manually enters year)
    input_year = st.number_input("📅 Enter Year for Prediction", min_value=2000, max_value=2100, value=2025, step=1)
    district = st.selectbox("🏙️ Select District", sorted(df["District"].unique()))

    # Prediction trigger button
    if st.button("🔮 Predict Crimes for Selected District & Year"):
        if district not in df["District"].values:
            st.error("⚠️ The entered district was not found in the dataset. Please check the spelling.")
            st.session_state.predicted_value = None
        else:
            X = df[features]
            y = df["Total Cognizable IPC crimes"]

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

            if model_choice == "Linear Regression":
                model = LinearRegression()
            elif model_choice == "Decision Tree":
                model = DecisionTreeRegressor(random_state=42)
            else:
                model = RandomForestRegressor(random_state=42)

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)

            st.success(f"✅ Model Trained Successfully using {model_choice}")
            st.write(f"**Mean Absolute Error:** {mae:.2f}")
            st.write(f"**R² Score:** {r2:.2f}")
            
            # Prediction for chosen district (for the given user-input year)
            selected_row = df[df["District"] == district][features]

            if not selected_row.empty:
                prediction = model.predict(selected_row)
                st.subheader(f"📍 Predicted Total Crimes in {district} for Year {input_year}: {prediction[0]:.0f}")
            else:
                st.warning("⚠️ No data found for the selected district.")