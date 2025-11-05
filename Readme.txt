## EduSense: Smart AI-Driven Classroom System

### **Abstract**

EduSense is an integrated Artificial Intelligence (AI) system designed to automate classroom management using real-time facial recognition, liveness verification, emotion analysis, and predictive analytics. The system records student attendance, detects spoofing attempts, monitors emotional well-being, and provides a Streamlit-powered dashboard that visualizes attendance data and predicts student dropout risks using machine learning. EduSense combines AI, computer vision, and data science to promote smarter, more empathetic education environments.

---

### **1. Introduction**

Traditional classroom attendance systems are manual and time-consuming. They are prone to human error, proxy attendance, and lack real insights into student engagement. EduSense bridges this gap by using facial recognition and emotion detection to create a fully automated attendance management and analytics solution.

Key objectives:

* Automate attendance marking using AI-driven facial recognition.
* Ensure authenticity using liveness detection (anti-spoofing).
* Detect and record emotional states for engagement insights.
* Provide a dashboard for visualization and dropout prediction.

---

### **2. System Architecture**

EduSense is divided into three main modules:

1. **Attendance Module (attendance.py)** – Real-time AI-based recognition.
2. **Dashboard Module (dashboard.py)** – Streamlit-based analytics and reports.
3. **Dropout Prediction Module (train_model.py)** – Predictive ML model for academic insights.

---

### **3. Module Descriptions**

#### **3.1 Attendance Module (attendance.py)**

This module performs real-time face detection, recognition, emotion analysis, and liveness verification.

**Key Features:**

* **Face Recognition:** Uses the `face_recognition` library to detect and match student faces.
* **Liveness Detection:** Implements the Local Binary Pattern (LBP) algorithm to differentiate real faces from spoofed ones.
* **Emotion Detection:** Employs DeepFace for emotion classification across categories such as Happy, Sad, Neutral, and Angry.
* **Alert System:** Displays visual alerts and logs when students appear consistently sad.
* **Data Logging:** Records attendance with timestamps and emotions into a date-specific CSV file.

**Output Example (CSV):**

| Name    | Time     | Emotion     |
| ------- | -------- | ----------- |
| Kavitha | 09:15:42 | Happy       |
| Neha    | 09:18:55 | Sad (Alert) |

---

#### **3.2 Dashboard Module (dashboard.py)**

The Streamlit-based dashboard acts as the central monitoring interface for the EduSense system.

**Functional Components:**

* **Dashboard Page:** Displays daily attendance summaries with metrics for Present, Absent, and Total Students.
* **Trend Visualization:** Generates hourly attendance trends using Plotly charts.
* **Attendance Sheet:** Displays real-time attendance logs and allows CSV downloads.
* **Webcam Mode:** Launches the real-time attendance system directly from the dashboard.
* **Dropout Prediction:** Integrates a pre-trained ML model for predicting student dropout risk.

**Error Handling:**

* Handles missing or malformed CSVs using `on_bad_lines='skip'`.
* Auto-adds missing columns (Name, Time, Emotion) when necessary.

---

#### **3.3 Dropout Prediction Module (train_model.py)**

This script trains and saves a predictive model that forecasts student dropout risk based on academic and socio-economic factors.

**Pipeline Components:**

* **Preprocessing:** Scales numerical data using `StandardScaler` and encodes categorical data using `OneHotEncoder`.
* **Balancing:** Utilizes `SMOTE` to correct class imbalances.
* **Model:** RandomForestClassifier trained on preprocessed data.
* **Model Persistence:** Saves the model as `dropout_model.pkl` using `dill` for dashboard integration.

**Key Dataset Columns:**

* Academic: `cgpa`, `attendance_rate`, `study_hours_per_week`
* Socio-Economic: `family_income`, `parental_education`, `scholarship`
* Target: `dropout`

---

### **4. Technical Stack**

| Component            | Technology Used                                        |
| -------------------- | ------------------------------------------------------ |
| Programming Language | Python 3.10+                                           |
| Computer Vision      | OpenCV, face-recognition                               |
| Emotion Analysis     | DeepFace (TensorFlow backend)                          |
| Dashboard            | Streamlit, Plotly                                      |
| ML Model             | scikit-learn, imbalanced-learn, RandomForestClassifier |
| Data Serialization   | dill                                                   |
| Data Handling        | pandas, numpy                                          |

---

### **5. Workflow Diagram**

**Step 1:** Capture webcam feed →
**Step 2:** Detect and encode faces →
**Step 3:** Verify liveness →
**Step 4:** Classify emotion →
**Step 5:** Log attendance in CSV →
**Step 6:** Streamlit dashboard visualizes and predicts insights.

---

### **6. Results and Discussion**

EduSense successfully automates the attendance process while improving data accuracy and reducing manual workload. The addition of liveness detection mitigates spoofing attempts, and emotion tracking provides valuable insights into student engagement levels. The dashboard visualizations enhance monitoring, and the dropout prediction model supports early intervention strategies.

---

### **7. Future Enhancements**

* Integrate emotion heatmaps for classroom engagement analysis.
* Send automated attendance and alert reports to faculty.
* Deploy lightweight versions for Raspberry Pi and edge devices.
* Integrate facial recognition with institution databases for seamless scalability.

---

### **8. Conclusion**

EduSense provides a comprehensive AI-based classroom management solution by combining face recognition, liveness detection, emotion tracking, and predictive analytics into one unified system. It exemplifies how AI can create smarter, safer, and more empathetic educational environments.

---

### **9. References**

* OpenCV Documentation: [https://opencv.org/](https://opencv.org/)
* DeepFace Library: [https://github.com/serengil/deepface](https://github.com/serengil/deepface)
* Streamlit Framework: [https://streamlit.io/](https://streamlit.io/)
* scikit-learn Documentation: [https://scikit-learn.org/](https://scikit-learn.org/)
* imbalanced-learn: [https://imbalanced-learn.org/](https://imbalanced-learn.org/)

---

### **Author**
Kavitha P Devasi
Mohamed Kaashif
Neha H R
Saayima Zuha
AI & ML Developer — EduSense Project
Department of Computer Science
