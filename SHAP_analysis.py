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
import os

# Create results directory if it doesn't exist
os.makedirs('results', exist_ok=True)

# Load the dataset
print("Loading Dataset_17_feat.xlsx...")
data = pd.read_excel('Dataset_17_feat.xlsx', engine='openpyxl')

# Define features based on actual columns in dataset
exclude_cols = ['Experimental_index', 'DP_Group', 'Time', 'T=0.25', 'T=0.5', 'T=1.0', 'Release']
features = [col for col in data.columns if col not in exclude_cols]
print("Using features:", features)

X = data[features]
y = (data['T=1.0'] < 0.2).astype(int)  # Changed threshold to 0.2

# Check class distribution
print("Class distribution (Slow vs. Fast/Intermediate):")
print(y.value_counts(normalize=True))

# Define groups and identify PLGA vs non-PLGA
groups = data['DP_Group']
# Identify PLGA based on LA/GA ratio presence
is_plga = data['LA/GA'].notna()

# Preprocess data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X.fillna(X.median()))
X_scaled_df = pd.DataFrame(X_scaled, columns=features)

# Apply SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_scaled, y)
print("Class distribution after SMOTE:")
print(pd.Series(y_resampled).value_counts(normalize=True))

# Adjust groups and PLGA identification for resampled data
groups_resampled = pd.Series(groups).repeat(X_resampled.shape[0] // len(groups) + 1)[:X_resampled.shape[0]]
is_plga_resampled = pd.Series(is_plga).repeat(X_resampled.shape[0] // len(is_plga) + 1)[:X_resampled.shape[0]]

# Use cross-validation
gkf = GroupKFold(n_splits=5)

# Step 1: Gradient Boosting Model (LightGBM)
lgb_params = {
    'objective': 'binary',
    'metric': 'binary_logloss',
    'boosting_type': 'gbdt',
    'num_leaves': 8,
    'learning_rate': 0.01,
    'feature_fraction': 0.6,
    'lambda_l1': 0.1,
    'lambda_l2': 0.1,
    'random_state': 42,
    'min_child_samples': 5
}

# Train-test split for each group
lgb_scores = {'roc_auc': [], 'f1': [], 'accuracy': []}
for train_idx, test_idx in gkf.split(X_resampled, y_resampled, groups_resampled):
    X_train, X_test = X_resampled[train_idx], X_resampled[test_idx]
    y_train, y_test = y_resampled[train_idx], y_resampled[test_idx]
    
    # Train LightGBM model
    lgb_train = lgb.Dataset(X_train, label=y_train)
    lgb_model = lgb.train(lgb_params, lgb_train, num_boost_round=20,
                         valid_sets=[lgb.Dataset(X_test, label=y_test)],
                         callbacks=[lgb.early_stopping(stopping_rounds=10, verbose=False)])
    
    # Make predictions
    y_pred = lgb_model.predict(X_test)
    y_pred_class = (y_pred > 0.5).astype(int)
    
    # Calculate metrics
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

# Step 2: Enhanced Separate SHAP Analysis for PLGA and Non-PLGA
explainer_lgb = shap.TreeExplainer(lgb_model)

# PLGA Entries
X_plga = X_resampled[is_plga_resampled]
shap_values_plga = explainer_lgb.shap_values(X_plga)

# Compute mean absolute SHAP values for PLGA
shap_importance_plga = np.abs(shap_values_plga).mean(axis=0)
shap_importance_plga_df = pd.DataFrame({
    'Feature': features,
    'Mean_Abs_SHAP': shap_importance_plga
}).sort_values(by='Mean_Abs_SHAP', ascending=False)

print("\nTop Features by SHAP Importance for PLGA Entries:")
print(shap_importance_plga_df)

# Bar plot for PLGA
plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values_plga, X_plga, feature_names=features, plot_type="bar", show=False)
plt.title("SHAP Feature Importance for PLGA Entries", fontsize=12)
plt.tight_layout()
plt.savefig('results/shap_summary_plga_bar.png', dpi=300, bbox_inches='tight')
plt.close()

# Beeswarm plot for PLGA
plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values_plga, X_plga, feature_names=features, show=False)
plt.title("SHAP Beeswarm Plot for PLGA Entries", fontsize=12)
plt.tight_layout()
plt.savefig('results/shap_summary_plga_beeswarm.png', dpi=300, bbox_inches='tight')
plt.close()

# Non-PLGA Entries
X_non_plga = X_resampled[~is_plga_resampled]
shap_values_non_plga = explainer_lgb.shap_values(X_non_plga)

# Compute mean absolute SHAP values for Non-PLGA
shap_importance_non_plga = np.abs(shap_values_non_plga).mean(axis=0)
shap_importance_non_plga_df = pd.DataFrame({
    'Feature': features,
    'Mean_Abs_SHAP': shap_importance_non_plga
}).sort_values(by='Mean_Abs_SHAP', ascending=False)

print("\nTop Features by SHAP Importance for Non-PLGA Entries:")
print(shap_importance_non_plga_df)

# Bar plot for Non-PLGA
plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values_non_plga, X_non_plga, feature_names=features, plot_type="bar", show=False)
plt.title("SHAP Feature Importance for Non-PLGA Entries", fontsize=12)
plt.tight_layout()
plt.savefig('results/shap_summary_non_plga_bar.png', dpi=300, bbox_inches='tight')
plt.close()

# Beeswarm plot for Non-PLGA
plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values_non_plga, X_non_plga, feature_names=features, show=False)
plt.title("SHAP Beeswarm Plot for Non-PLGA Entries", fontsize=12)
plt.tight_layout()
plt.savefig('results/shap_summary_non_plga_beeswarm.png', dpi=300, bbox_inches='tight')
plt.close()

# Compare feature importance between PLGA and Non-PLGA
comparison_df = pd.DataFrame({
    'Feature': features,
    'PLGA_SHAP': shap_importance_plga,
    'Non_PLGA_SHAP': shap_importance_non_plga
})
comparison_df['Difference'] = comparison_df['PLGA_SHAP'] - comparison_df['Non_PLGA_SHAP']
comparison_df = comparison_df.sort_values('Difference', key=abs, ascending=False)

# Plot feature importance comparison
plt.figure(figsize=(12, 8))
comparison_df.plot(x='Feature', y=['PLGA_SHAP', 'Non_PLGA_SHAP'], kind='bar')
plt.title('Feature Importance Comparison: PLGA vs Non-PLGA')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('results/feature_importance_comparison.png', dpi=300, bbox_inches='tight')
plt.close()

# Save comparison results
comparison_df.to_csv('results/feature_importance_comparison.csv', index=False)
print("\nAnalysis completed. Results saved in the 'results' directory.")