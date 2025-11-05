import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import os
import subprocess
from datetime import datetime
import dill  # <-- For loading the .pkl model

# ---------------- CONFIG ----------------
st.set_page_config(
    page_title="EduSense | Smart Attendance Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

CSS_FILE = "style.css"

# ---------------- LOAD CUSTOM CSS ----------------
if os.path.exists(CSS_FILE):
    with open(CSS_FILE) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
else:
    st.warning(f"CSS file not found. Create a '{CSS_FILE}' file for custom styling.")

# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.title("🎓 EduSense")
    st.markdown("### 📍 Navigation")
    pages = ["Dashboard", "Attendance Sheet", "Webcam Mode", "Reports", "Students Dropout Prediction"]
    choice = st.radio("Go to:", pages, index=0, label_visibility="collapsed")
    st.divider()
    st.caption("Built with ❤️ using Streamlit + OpenCV")

# ---------------- DASHBOARD PAGE ----------------
if choice == "Dashboard":
    st.subheader("📊 Attendance Overview")

    col1, col2, col3 = st.columns(3)
    col1.metric("Present | Today", "145", "↑ 12%")
    col2.metric("Absent | Today", "32", "↓ 5%")
    col3.metric("Attendance | This Month", "87%", "↑ 4%")

    st.markdown("---")

    st.subheader("📈 Daily Attendance Trend")
    time = ["08:00", "09:00", "10:00", "11:00", "12:00", "01:00", "02:00"]
    present = [20, 30, 40, 25, 60, 80, 65]
    absent = [5, 10, 12, 8, 7, 5, 9]
    total = [25, 40, 52, 33, 67, 85, 74]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=time, y=present, mode='lines+markers', name='Present', line=dict(color='#3B82F6', width=3)))
    fig.add_trace(go.Scatter(x=time, y=absent, mode='lines+markers', name='Absent', line=dict(color='#EF4444', width=3)))
    fig.add_trace(go.Scatter(x=time, y=total, mode='lines+markers', name='Total Students', line=dict(color='#A0AEC0', width=2, dash='dot')))

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="#F0F2F6",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis=dict(gridcolor='#3E404C'),
        yaxis=dict(gridcolor='#3E404C')
    )
    st.plotly_chart(fig, use_container_width=True)

# ---------------- ATTENDANCE SHEET ----------------
elif choice == "Attendance Sheet":
    st.subheader("📋 Attendance Records")

    today = datetime.now().strftime("%Y-%m-%d")
    file_name = f"{today}_attendance.csv"

    if os.path.exists(file_name):
        try:
            df = pd.read_csv(file_name)
            st.dataframe(df, use_container_width=True)

            col1, col2 = st.columns([1, 3])
            with col1:
                st.download_button(
                    "⬇️ Download Attendance CSV",
                    df.to_csv(index=False).encode(),
                    f"{file_name}"
                )
        except pd.errors.EmptyDataError:
            st.warning(f"⚠️ The attendance file '{file_name}' is empty.")
        except Exception as e:
            st.error(f"❌ Error reading file '{file_name}': {e}")
    else:
        st.info(f"No attendance file found for today ('{file_name}').")
        st.write("Run the 'Webcam Mode' to start generating records.")

# ---------------- WEBCAM MODE ----------------
elif choice == "Webcam Mode":
    st.subheader("🎥 Real-Time Attendance System")
    st.markdown("Click the button below to launch the camera-based attendance system in a new window.")

    if st.button("🚀 Start Real-Time Attendance"):
        try:
            script_to_run = "attendance.py"
            subprocess.Popen(["python", script_to_run])
            st.success(f"✅ Started '{script_to_run}'! Check the new camera window.")
            st.info("Press 'q' in the camera window to stop.")
        except FileNotFoundError:
            st.error(f"⚠️ Error: '{script_to_run}' not found.")
            st.info(f"Make sure the script is in the same directory as this dashboard.")
        except Exception as e:
            st.error(f"⚠️ Error running {script_to_run}: {e}")

# ---------------- REPORTS ----------------
elif choice == "Reports":
    st.subheader("📊 Reports & Insights")
    st.info("Future section: analytics and performance insights coming soon...")

# ---------------- STUDENTS DROPOUT PREDICTION ----------------
elif choice == "Students Dropout Prediction":
    st.subheader("🔮 Student Dropout Prediction")
    st.info("This section uses a pre-trained model to predict student dropout risk based on their data.")
    st.divider()

    MODEL_PATH = "dropout_model.pkl"
    DATA_PATH = "cleaned_data.csv"

    # --- 1. Load Model (cached) ---
    @st.cache_resource
    def load_model():
        if not os.path.exists(MODEL_PATH):
            st.error(f"❌ Error: Model file not found. Make sure '{MODEL_PATH}' is in the same folder.")
            return None
        try:
            with open(MODEL_PATH, "rb") as f:
                model = dill.load(f)
            return model
        except Exception as e:
            st.error(f"❌ Error loading model: {e}")
            return None

    # --- 2. Load Student Data (cached) ---
    @st.cache_data
    def load_data():
        if not os.path.exists(DATA_PATH):
            st.error(f"❌ Error: Student data file not found. Make sure '{DATA_PATH}' is in the same folder.")
            return pd.DataFrame()
        try:
            df = pd.read_csv(DATA_PATH)
            if "student_id" not in df.columns:
                st.error(f"❌ '{DATA_PATH}' must contain a 'student_id' column.")
                return pd.DataFrame()
            df['student_id'] = df['student_id'].astype(str)
            return df
        except Exception as e:
            st.error(f"❌ Error loading data: {e}")
            return pd.DataFrame()

    # --- 3. Define Prediction Function ---
    def predict_dropout(model, features):
        try:
            prediction = model.predict(features)[0]
            probabilities = model.predict_proba(features)[0]
            if prediction == 1:
                prob = probabilities[1]
                return f"Prediction: **High Risk of Dropout** (Confidence: {prob*100:.2f}%)"
            else:
                prob = probabilities[0]
                return f"Prediction: **Low Risk of Dropout** (Confidence: {prob*100:.2f}%)"
        except Exception as e:
            st.error(f"Error during prediction: {e}")
            if hasattr(model, "feature_names_in_"):
                st.error(f"Model feature names: {list(model.feature_names_in_)}")
            return None

    # --- Load model and data ---
    model = load_model()
    df = load_data()

    # --- 4. Build UI ---
    if model is not None and not df.empty:
        col1, col2 = st.columns([1, 2])

        with col1:
            st.subheader("🔍 Check Single Student")
            student_id_input = st.text_input("Enter Student ID:", placeholder="e.g., 12345").strip()

            if st.button("Predict Risk"):
                if not student_id_input:
                    st.warning("Please enter a Student ID.")
                else:
                    student_row = df[df["student_id"] == student_id_input]
                    if student_row.empty:
                        st.warning(f"Student ID '{student_id_input}' not found in the data.")
                    else:
                        student_features = student_row.copy()
                        missing_cols = []
                        for col in model.feature_names_in_:
                            if col not in student_features.columns:
                                student_features[col] = 0
                                missing_cols.append(col)
                        try:
                            student_features_final = student_features[model.feature_names_in_]
                            result = predict_dropout(model, student_features_final)
                            if result:
                                if "High" in result:
                                    st.error(result)
                                else:
                                    st.success(result)
                            st.write(f"**Details for Student: {student_id_input}**")
                            st.dataframe(student_row, use_container_width=True)
                            if missing_cols:
                                st.info(f"Note: Missing features set to 0: {', '.join(missing_cols)}")
                        except Exception as e:
                            st.error(f"Error during prediction: {e}")

        with col2:
            st.subheader("📚 Full Student List")
            search = st.text_input("Search full list:", placeholder="Search by ID, department, etc...").lower()

            if search:
                filtered_df = df[df.apply(lambda row: search in row.astype(str).to_string().lower(), axis=1)]
            else:
                filtered_df = df

            st.dataframe(filtered_df, use_container_width=True, height=500)

    elif model is None:
        st.info("Model could not be loaded. Please check the file path and logs.")
    elif df.empty:
        st.info("Student data could not be loaded. Please check the file path and logs.")
