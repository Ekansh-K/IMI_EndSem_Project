import pandas as pd
import numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
import lightgbm as lgb
import tensorflow as tf
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.svm import SVC
import shap
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE

# Load the updated dataset
data = pd.read_csv('updated_lai_dataset_final.csv')

# Features and target
features = ['Drug_Mw', 'Log_Polymer_MW', 'Log_DLC', 'Polymer_Hydrophobicity', 
            'Polymer_Tg', 'Degradation_Rate', 'SA_V', 'Drug_logP', 'Drug_TPSA', 
            'Drug_NHA', 'Drug_Solubility', 'Polymer_Drug_Compatibility', 
            'Release_Rate_Slope', 'Drug_RotatableBonds', 'Drug_HBD', 
            'Polymer_Hydrophobicity_Drug_logP', 'Drug_Solubility_DLC']
available_features = [f for f in features if f in data.columns]
print("Using features:", available_features)

X = data[available_features]
y = (data['T=1.0'] < 0.25).astype(int)

# Check class distribution
print("Class distribution (Slow vs. Fast/Intermediate):")
print(y.value_counts(normalize=True))

# Define DP_Group and Polymer_Type for splitting
groups = data['DP_Group']
polymer_types = data['Polymer_Type']

# Preprocess data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled_df = pd.DataFrame(X_scaled, columns=available_features)

# Apply SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_scaled, y)
print("Class distribution after SMOTE:")
print(pd.Series(y_resampled).value_counts(normalize=True))

# Adjust groups and polymer_types for resampled data
groups_resampled = pd.Series(groups).repeat(X_resampled.shape[0] // len(groups) + 1)[:X_resampled.shape[0]]
polymer_types_resampled = pd.Series(polymer_types).repeat(X_resampled.shape[0] // len(polymer_types) + 1)[:X_resampled.shape[0]]

# Use cross-validation
gkf = GroupKFold(n_splits=5)

# Step 1: Gradient Boosting Model (LightGBM)
lgb_params = {
    'objective': 'binary',
    'metric': 'binary_logloss',
    'boosting_type': 'gbdt',
    'num_leaves': 8,  # Reduced further
    'learning_rate': 0.01,  # Slower learning
    'feature_fraction': 0.6,
    'lambda_l1': 0.1,  # L1 regularization
    'lambda_l2': 0.1,  # L2 regularization
    'random_state': 42,
    'min_child_samples': 5
}

lgb_scores = {'roc_auc': [], 'f1': [], 'accuracy': []}
for train_idx, test_idx in gkf.split(X_resampled, y_resampled, groups_resampled):
    X_train, X_test = X_resampled[train_idx], X_resampled[test_idx]
    y_train, y_test = y_resampled[train_idx], y_resampled[test_idx]
    lgb_train = lgb.Dataset(X_train, label=y_train)
    lgb_model = lgb.train(lgb_params, lgb_train, num_boost_round=20,
                          valid_sets=[lgb.Dataset(X_test, label=y_test)],
                          callbacks=[lgb.early_stopping(stopping_rounds=10, verbose=False)])
    y_pred = lgb_model.predict(X_test)
    y_pred_class = (y_pred > 0.5).astype(int)
    lgb_scores['roc_auc'].append(roc_auc_score(y_test, y_pred))
    lgb_scores['f1'].append(classification_report(y_test, y_pred_class, output_dict=True)['1']['f1-score'])
    lgb_scores['accuracy'].append((y_pred_class == y_test).mean())
    print("\nConfusion Matrix (Fold):")
    print(confusion_matrix(y_test, y_pred_class))

print("\nLightGBM Cross-Validation Results:")
print(f"ROC-AUC: {np.mean(lgb_scores['roc_auc']):.3f} (+/- {np.std(lgb_scores['roc_auc']) * 2:.3f})")
print(f"F1-Score (Slow): {np.mean(lgb_scores['f1']):.3f} (+/- {np.std(lgb_scores['f1']) * 2:.3f})")
print(f"Accuracy: {np.mean(lgb_scores['accuracy']):.3f} (+/- {np.std(lgb_scores['accuracy']) * 2:.3f})")

# Train on full data for SHAP
lgb_train_full = lgb.Dataset(X_resampled, label=y_resampled)
lgb_model = lgb.train(lgb_params, lgb_train_full, num_boost_round=20)

# Step 2: Separate SHAP Analysis for PLGA and Non-PLGA
explainer_lgb = shap.TreeExplainer(lgb_model)

# PLGA entries
plga_mask = (polymer_types_resampled == 'PLGA')
X_plga = X_resampled[plga_mask]
shap_values_plga = explainer_lgb.shap_values(X_plga)
print("\nSHAP Summary for PLGA Entries:")
shap.summary_plot(shap_values_plga, X_plga, feature_names=available_features, plot_type="bar")
plt.savefig('shap_summary_lgb_plga.png')
plt.close()

# Non-PLGA entries
non_plga_mask = (polymer_types_resampled != 'PLGA')
X_non_plga = X_resampled[non_plga_mask]
shap_values_non_plga = explainer_lgb.shap_values(X_non_plga)
print("\nSHAP Summary for Non-PLGA Entries:")
shap.summary_plot(shap_values_non_plga, X_non_plga, feature_names=available_features, plot_type="bar")
plt.savefig('shap_summary_lgb_non_plga.png')
plt.close()

# Step 3: Neural Network Model (Simplified for Small Dataset)
nn_scores = {'roc_auc': [], 'f1': [], 'accuracy': []}
for train_idx, test_idx in gkf.split(X_resampled, y_resampled, groups_resampled):
    X_train, X_test = X_resampled[train_idx], X_resampled[test_idx]
    y_train, y_test = y_resampled[train_idx], y_resampled[test_idx]
    nn_model = tf.keras.Sequential([
        tf.keras.layers.Dense(12, activation='relu', input_dim=X_train.shape[1]),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(6, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    nn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    nn_model.fit(X_train, y_train, epochs=15, batch_size=16, verbose=0, validation_data=(X_test, y_test))
    y_pred = nn_model.predict(X_test, verbose=0).flatten()
    y_pred_class = (y_pred > 0.5).astype(int)
    nn_scores['roc_auc'].append(roc_auc_score(y_test, y_pred))
    nn_scores['f1'].append(classification_report(y_test, y_pred_class, output_dict=True)['1']['f1-score'])
    nn_scores['accuracy'].append((y_pred_class == y_test).mean())

print("\nNeural Network Cross-Validation Results:")
print(f"ROC-AUC: {np.mean(nn_scores['roc_auc']):.3f} (+/- {np.std(nn_scores['roc_auc']) * 2:.3f})")
print(f"F1-Score (Slow): {np.mean(nn_scores['f1']):.3f} (+/- {np.std(nn_scores['f1']) * 2:.3f})")
print(f"Accuracy: {np.mean(nn_scores['accuracy']):.3f} (+/- {np.std(nn_scores['accuracy']) * 2:.3f})")

# Step 4: Other Models (for Comparison)
models = {
    'RandomForest': RandomForestClassifier(n_estimators=20, max_depth=3, random_state=42),
    'XGBoost': xgb.XGBClassifier(n_estimators=20, learning_rate=0.01, max_depth=3, random_state=42),
    'SVM': SVC(probability=True, kernel='rbf', random_state=42),
    'LightGBM': lgb.LGBMClassifier(**lgb_params)
}

scoring = {'roc_auc': 'roc_auc', 'f1': 'f1', 'accuracy': 'accuracy'}
for name, model in models.items():
    print(f"\n{name} Cross-Validation Results:")
    scores = {'roc_auc': [], 'f1': [], 'accuracy': []}
    for train_idx, test_idx in gkf.split(X_resampled, y_resampled, groups_resampled):
        X_train, X_test = X_resampled[train_idx], X_resampled[test_idx]
        y_train, y_test = y_resampled[train_idx], y_resampled[test_idx]
        model.fit(X_train, y_train)
        y_pred = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else model.predict(X_test)
        y_pred_class = (y_pred > 0.5).astype(int)
        scores['roc_auc'].append(roc_auc_score(y_test, y_pred))
        scores['f1'].append(classification_report(y_test, y_pred_class, output_dict=True)['1']['f1-score'])
        scores['accuracy'].append((y_pred_class == y_test).mean())
    print(f"ROC-AUC: {np.mean(scores['roc_auc']):.3f} (+/- {np.std(scores['roc_auc']) * 2:.3f})")
    print(f"F1-Score: {np.mean(scores['f1']):.3f} (+/- {np.std(scores['f1']) * 2:.3f})")
    print(f"Accuracy: {np.mean(scores['accuracy']):.3f} (+/- {np.std(scores['accuracy']) * 2:.3f})")