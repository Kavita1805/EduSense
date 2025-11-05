import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import os
import subprocess
from datetime import datetime, timedelta
import dill
import time

# ---------------- CONFIG ----------------
st.set_page_config(
    page_title="EduSense | Smart Attendance Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- File Constants ---
CSS_FILE = "style.css"
TODAY_DATE = datetime.now()
YESTERDAY_DATE = TODAY_DATE - timedelta(days=1)
ATTENDANCE_FILE = f"{TODAY_DATE.strftime('%Y-%m-%d')}_attendance.csv"
YESTERDAY_ATTENDANCE_FILE = f"{YESTERDAY_DATE.strftime('%Y-%m-%d')}_attendance.csv"
MASTER_LIST_FILE = "cleaned_data.csv"

# ---------------- LOAD CUSTOM CSS ----------------
if os.path.exists(CSS_FILE):
    with open(CSS_FILE) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
else:
    st.warning(f"CSS file not found. Create a '{CSS_FILE}' file for custom styling.")


# --- ⭐ MODIFIED Helper function ---
@st.cache_data
def load_attendance_data(file_name):
    if not os.path.exists(file_name):
        return pd.DataFrame(columns=["Name", "Time", "Emotion"])
    try:
        # --- FIX: Added on_bad_lines='skip' to ignore malformed lines ---
        df = pd.read_csv(file_name, on_bad_lines='skip')
        
        if 'Time' not in df.columns and 'Name' not in df.columns:
            # Also add it here for the headerless case
            df = pd.read_csv(file_name, header=None, names=["Name", "Time", "Emotion"], on_bad_lines='skip')
        
        # Ensure correct columns exist even if file is weird
        if "Name" not in df.columns: df["Name"] = None
        if "Time" not in df.columns: df["Time"] = None
        if "Emotion" not in df.columns: df["Emotion"] = None
            
        return df
        
    except pd.errors.EmptyDataError:
        return pd.DataFrame(columns=["Name", "Time", "Emotion"])
    except Exception as e:
        st.error(f"Error loading {file_name}: {e}")
        return pd.DataFrame(columns=["Name", "Time", "Emotion"])
# --- ⭐ END OF MODIFIED SECTION ---


# --- Helper function to get total student count ---
@st.cache_resource
def get_total_student_count(file_name):
    if not os.path.exists(file_name):
        st.warning(f"Master student file '{file_name}' not found. 'Total' and 'Absent' counts will be 0.")
        return 0
    try:
        df = pd.read_csv(file_name)
        return len(df)
    except Exception as e:
        st.error(f"Error loading master list {file_name}: {e}")
        return 0

# --- Helper function to format delta percentages ---
def format_delta_percent(today_val, yesterday_val):
    if yesterday_val == 0:
        if today_val > 0:
            return "↑ New"
        else:
            return None
    
    delta_percent = ((today_val - yesterday_val) / yesterday_val) * 100
    
    if delta_percent > 0:
        return f"↑ {delta_percent:.0f}%"
    elif delta_percent < 0:
        return f"↓ {abs(delta_percent):.0f}%"
    else:
        return "0%"

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
    
    # Load data
    today_attendance_df = load_attendance_data(ATTENDANCE_FILE)
    yesterday_attendance_df = load_attendance_data(YESTERDAY_ATTENDANCE_FILE)
    total_students = get_total_student_count(MASTER_LIST_FILE)
    
    # Calculate today's metrics
    present_count = today_attendance_df['Name'].nunique()
    absent_count = total_students - present_count
    
    # Calculate yesterday's metrics
    yesterday_present_count = yesterday_attendance_df['Name'].nunique()
    yesterday_absent_count = total_students - yesterday_present_count
    
    # Get delta strings
    present_delta_str = format_delta_percent(present_count, yesterday_present_count)
    absent_delta_str = format_delta_percent(absent_count, yesterday_absent_count)

    col1, col2, col3 = st.columns(3)
    col1.metric("Present | Today", f"{present_count}", delta=present_delta_str, delta_color="normal")
    col2.metric("Absent | Today", f"{absent_count}", delta=absent_delta_str, delta_color="inverse")
    col3.metric("Total Enrolled", f"{total_students}")

    st.markdown("---")
    
    st.subheader("📈 Daily Attendance Trend (Live from CSV)")
    
    if today_attendance_df.empty or 'Time' not in today_attendance_df.columns or today_attendance_df['Time'].isnull().all():
        if not os.path.exists(ATTENDANCE_FILE):
             st.info(f"No attendance file found for today ('{ATTENDANCE_FILE}').")
             st.write("Run the 'Webcam Mode' to start generating attendance data.")
        else:
            st.info(f"Attendance file '{ATTENDANCE_FILE}' is empty or contains no valid time data. No data to plot yet.")
    else:
        try:
            # Process the data
            df_plot = today_attendance_df.copy().dropna(subset=['Time']) # Drop rows where Time is missing
            df_plot['Hour'] = pd.to_datetime(df_plot['Time'], format='%H:%M:%S').dt.hour
            hourly_counts = df_plot.groupby('Hour')['Name'].count().reset_index().rename(columns={'Name': 'Present'})
            
            all_hours = pd.DataFrame({'Hour': range(7, 19)}) # 7:00 to 18:00
            hourly_data = pd.merge(all_hours, hourly_counts, on='Hour', how='left').fillna(0)
            hourly_data['Present'] = hourly_data['Present'].astype(int)
            hourly_data['Hour_str'] = hourly_data['Hour'].apply(lambda h: f"{h:02d}:00")

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=hourly_data['Hour_str'], 
                y=hourly_data['Present'], 
                mode='lines+markers', 
                name='Present Students Marked', 
                line=dict(color='#3B82F6', width=3)
            ))

            fig.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font_color="#F0F2F6",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                xaxis=dict(gridcolor='#3E404C', title='Hour of Day'),
                yaxis=dict(gridcolor='#3E404C', title='Students Marked Present')
            )
            
            st.plotly_chart(fig, use_container_width=True)
            st.markdown("ℹ️ *This chart shows the number of students marked present **during** each hour, based on today's attendance file.*")

        except Exception as e:
            st.error(f"❌ Error processing chart data: {e}")

    # Auto-refresh loop
    time.sleep(5)
    st.rerun()


# ---------------- ATTENDANCE SHEET ----------------
elif choice == "Attendance Sheet":
    st.subheader("📋 Attendance Records")

    df_attendance = load_attendance_data(ATTENDANCE_FILE)
    
    if not os.path.exists(ATTENDANCE_FILE):
         st.info(f"No attendance file found for today ('{ATTENDANCE_FILE}').")
         st.write("Run the 'Webcam Mode' to start generating records.")
    elif df_attendance.empty:
        st.warning(f"⚠️ The attendance file '{ATTENDANCE_FILE}' is empty.")
    else:
        st.dataframe(df_attendance, use_container_width=True)
        
        col1, col2 = st.columns([1, 3])
        with col1:
            st.download_button(
                "⬇️ Download Attendance CSV",
                df_attendance.to_csv(index=False).encode(),
                f"{ATTENDANCE_FILE}"
            )

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