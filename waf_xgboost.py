 waf_xgboost_improved.py
import pandas as pd
import numpy as np
import re
from urllib.parse import urlparse
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report, precision_recall_curve
)
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
import joblib
import warnings
warnings.filterwarnings("ignore")

#%%
print("📥 Loading dataset...")
data = pd.read_csv('/kaggle/input/cac-da-waf-synthetic-cisc-ecml/CAC_DA_WAF_Synthetic_CISC_ECML_logs_20251114.csv')
print(f"✅ Loaded dataset with shape: {data.shape}")
#%%
# to display the whole columns
pd.set_option('display.max_columns', None)
#%%
# -----------------------------
# 🔧 Feature Engineering
# -----------------------------
print("🔧 Performing feature engineering...")

data['URI_length'] = data['URI'].apply(lambda x: len(str(x)))
data['digit_ratio'] = data['URI'].apply(lambda x: sum(c.isdigit() for c in str(x)) / len(str(x)) if len(str(x)) > 0 else 0)
data['entropy'] = data['URI'].apply(
    lambda x: -sum((str(x).count(c) / len(str(x))) * np.log2(str(x).count(c) / len(str(x))) for c in set(str(x)))
    if len(str(x)) > 0 else 0
)
data['special_char_count'] = data['URI'].apply(lambda x: sum(not c.isalnum() for c in str(x)))
data['suspicious_keyword'] = data['URI'].apply(lambda x: 1 if 'suspicious' in str(x).lower() else 0)
data['uppercase_ratio'] = data['URI'].apply(lambda x: sum(c.isupper() for c in str(x)) / len(str(x)) if len(str(x)) > 0 else 0)
data['unique_char_ratio'] = data['URI'].apply(lambda x: len(set(str(x))) / len(str(x)) if len(str(x)) > 0 else 0)
data['query_param_count'] = data['URI'].apply(lambda x: str(x).count('='))
data.drop(columns=['timestamp', 'ID'], inplace=True, errors='ignore')


#%%
data
#%%
# -----------------------------
# 🔠 Encode Categorical Columns
# -----------------------------
categorical_cols = data.select_dtypes(include=['object']).columns
print(f" Encoding {len(categorical_cols)} categorical columns.")

label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col].astype(str))
    label_encoders[col] = le
#%%
data
#%%
# checking whether the data is imbalanced or not
print(" Class distribution before balancing:")
print(data['Class'].value_counts())
#%%
# -----------------------------
# Train-Test Split
# -----------------------------
x = data.drop(columns=['Class'])
y = data['Class']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.15, random_state=42, stratify=y)

print(f"Final dataset shape for training: {x_train.shape}")
#%%
import numpy as np
from sklearn.utils.class_weight import compute_sample_weight

# Calculate weights for each sample
sample_weights = compute_sample_weight(
    class_weight='balanced',
    y=y_train
)

#%%
print(data['Class'].value_counts())
#%%
# Create DMatrix with weight

dtrain = xgb.DMatrix(x_train, label=y_train, weight=sample_weights)
#%%
# -----------------------------
# ⚙ XGBoost Model
# -----------------------------

params = {
    'objective': 'binary:logistic',
    'learning_rate': 0.05,
    'max_depth': 7,
    'n_estimators': 400,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'eval_metric': 'logloss',
    'use_label_encoder': False,
   # 'scale_pos_weight': 1,  # balanced dataset, no need to upweight
}
#%%
print(" Training improved XGBoost model...")

model = xgb.XGBClassifier(**params)
model.fit(x_train, y_train, sample_weight=sample_weights)

print(" Model training completed.")
#%%
# Initialize and train the XGBoost classifier
xgb_model = xgb.XGBClassifier()
#%%

#%%
data
#%%
# -----------------------------
#  Model Evaluation
# -----------------------------
y_prob = model.predict_proba(x_test)[:, 1]
threshold = 0.35
y_pred = (y_prob > threshold).astype(int)

accuracy = accuracy_score(y_test,y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_prob)
conf_matrix = confusion_matrix(y_test, y_pred)

print("\n===== 🧮 MODEL PERFORMANCE =====")
print(f"Accuracy     : {accuracy:.4f}")
print(f"Precision    : {precision:.4f}")
print(f"Recall       : {recall:.4f}")
print(f"F1 Score     : {f1:.4f}")
print(f"ROC–AUC Score: {roc_auc:.4f}")
print("\nConfusion Matrix:")
print(conf_matrix)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))


#%%
# -----------------------------
#  Visualization
# -----------------------------
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Normal', 'Anomaly'], yticklabels=['Normal', 'Anomaly'])
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title('Confusion Matrix — Improved XGBoost')
plt.show()


#%%
# Feature importance
plt.figure(figsize=(10, 8))
xgb.plot_importance(model, max_num_features=15)
plt.title('Top 15 Most Important Features')
plt.show()

print("\n Model ready for deployment — balanced and optimized for anomaly detection.")
