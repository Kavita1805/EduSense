import pandas as pd
import dill
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline as SklearnPipeline
from imblearn.pipeline import Pipeline as ImblearnPipeline
from imblearn.over_sampling import SMOTE
import os

print("--- Model Training Script Started ---")

# --- 1. Load Data ---
DATA_FILE = "cleaned_data.csv"
MODEL_FILE = "dropout_model.pkl"

if not os.path.exists(DATA_FILE):
    print(f"❌ ERROR: File not found: '{DATA_FILE}'. Please make sure it's in the same folder.")
    exit()

try:
    df = pd.read_csv(DATA_FILE)
    print(f"✅ Successfully loaded '{DATA_FILE}'. Shape: {df.shape}")
except Exception as e:
    print(f"❌ ERROR: Could not read file: {e}")
    exit()


# --- 2. CONFIGURATION (Based on your CSV headers) ---

# This is the column we are trying to predict
YOUR_TARGET_COLUMN = "dropout"

# List of all numeric feature columns from your CSV
numeric_features = [
    'age', 
    'cgpa', 
    'attendance_rate', 
    'family_income', 
    'past_failures', 
    'study_hours_per_week', 
    'assignments_submitted', 
    'projects_completed', 
    'total_activities'
]

# List of all categorical feature columns from your CSV
categorical_features = [
    'gender', 
    'department', 
    'scholarship', 
    'parental_education', 
    'extra_curricular',
    'sports_participation'
]
# Note: 'student_id' is correctly excluded from training.

# --- End of Configuration ---


# --- 3. Define Features (X) and Target (y) ---
try:
    # Check if target column exists
    if YOUR_TARGET_COLUMN not in df.columns:
        print(f"❌ ERROR: The target column '{YOUR_TARGET_COLUMN}' was not found in the CSV.")
        print(f"Available columns are: {list(df.columns)}")
        exit()
        
    y = df[YOUR_TARGET_COLUMN]

    # Check and prepare feature columns
    all_features_list = numeric_features + categorical_features
    features_in_df = []
    features_missing = []
    
    for col in all_features_list:
        if col in df.columns:
            features_in_df.append(col)
        else:
            features_missing.append(col)
            
    if features_missing:
        print(f"⚠️ Warning: The following feature columns were not found in the CSV and will be ignored:")
        print(f"   {features_missing}")

    X = df[features_in_df]
    print(f"✅ Defined features (X) and target (y). Using {len(features_in_df)} features.")

except Exception as e:
    print(f"❌ ERROR: Failed to define X and y. Error: {e}")
    exit()


# --- 4. Create Preprocessing Pipelines ---

# Pipeline for numeric data: Scale it
numeric_transformer = SklearnPipeline(steps=[
    ('scaler', StandardScaler())
])

# Pipeline for categorical data: One-Hot Encode it
categorical_transformer = SklearnPipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

# Combine both pipelines using ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ],
    remainder='drop' # Drop any columns not in our lists
)
print("✅ Created preprocessing pipeline.")


# --- 5. Create the Full Model Pipeline (Preprocess -> SMOTE -> Model) ---

# This pipeline will:
# 1. Apply the 'preprocessor' (scaling and one-hot encoding)
# 2. Apply 'smote' to fix class imbalance (only on training data)
# 3. Train the 'model' (Random Forest)

pipeline = ImblearnPipeline(steps=[
    ('preprocessor', preprocessor),
    ('smote', SMOTE(random_state=42)),
    ('model', RandomForestClassifier(random_state=42, n_estimators=100))
])

print("✅ Created full model pipeline with SMOTE and RandomForest.")

# --- 6. Split and Train the Model ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"Splitting data: {len(X_train)} training samples, {len(X_test)} testing samples.")
print("Starting model training... (This may take a moment)")

pipeline.fit(X_train, y_train)

print("✅ Model training complete!")

# --- 7. Save the Model ---
try:
    with open(MODEL_FILE, "wb") as f:
        dill.dump(pipeline, f)
    print(f"✅ Model successfully saved to '{MODEL_FILE}'!")
    print("   You can now run your Streamlit app.")
except Exception as e:
    print(f"❌ ERROR: Could not save model: {e}")

print("--- Model Training Script Finished ---")