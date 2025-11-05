import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import os
import subprocess
from datetime import datetime

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

# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.title("🎓 EduSense Dashboard")
    st.markdown("### 📍 Navigation")
    pages = ["Dashboard", "Attendance Sheet", "Webcam Mode", "Reports", "Students"]
    choice = st.radio("Go to:", pages)
    st.divider()
    st.caption("Built with ❤️ using Streamlit + OpenCV + Face Recognition")

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
    fig.add_trace(go.Scatter(x=time, y=present, mode='lines+markers', name='Present'))
    fig.add_trace(go.Scatter(x=time, y=absent, mode='lines+markers', name='Absent'))
    fig.add_trace(go.Scatter(x=time, y=total, mode='lines+markers', name='Total Students'))
    fig.update_layout(template="plotly_dark", paper_bgcolor="#040000", plot_bgcolor="#040000", font_color="#c6c5c5")
    st.plotly_chart(fig, use_container_width=True)

# ---------------- ATTENDANCE SHEET ----------------
elif choice == "Attendance Sheet":
    st.subheader("📋 Attendance Records")

    today = datetime.now().strftime("%Y-%m-%d")
    file_name = f"{today}.csv"

    if os.path.exists(file_name):
        df = pd.read_csv(file_name)
        st.dataframe(df, use_container_width=True)
        st.download_button(
            "⬇️ Download Attendance CSV",
            df.to_csv(index=False).encode(),
            f"{file_name}"
        )
    else:
        st.info("No attendance records found for today yet.")

# ---------------- WEBCAM MODE ----------------
elif choice == "Webcam Mode":
    st.subheader("🎥 Real-Time Attendance System")
    st.markdown("Click the button below to launch the camera-based attendance system (LBP + liveness).")

    if st.button("Start Attendance"):
        try:
            subprocess.Popen(["python", "attendance.py"])
            st.success("✅ Attendance system started! Check the camera window.")
            st.info("Press 'Q' in the camera window to stop.")
        except Exception as e:
            st.error(f"⚠️ Error running attendance.py: {e}")

# ---------------- REPORTS ----------------
elif choice == "Reports":
    st.subheader("📊 Reports & Insights")
    st.info("Future section: analytics and performance insights coming soon...")

# ---------------- STUDENTS ----------------
elif choice == "Students":
    st.subheader("👥 Manage Students")
    st.info("Add or remove student image files in this folder to update the system.")
