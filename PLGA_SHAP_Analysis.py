import pandas as pd
import numpy as np
import lightgbm as lgb
import shap
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Read the dataset
df = pd.read_csv('updated_lai_dataset_enhanced.csv')

# Separate features and target
features = ['DLC', 'SA-V', 'Drug_Tm', 'Drug_Pka', 'Initial D/M ratio', 
           'Drug_Mw', 'Drug_TPSA', 'Drug_NHA', 'Drug_LogP', 'Time',
           'Polymer_MW', 'Polymer_Hydrophobicity', 'Polymer_Tg', 
           'Degradation_Rate', 'Polymer_Drug_Compatibility']

X = df[features]
y = df['Release']

# Create PLGA and non-PLGA masks
plga_mask = df['DP_Group'].str.contains('PLGA')
non_plga_mask = ~plga_mask

# Function to train model and get SHAP values
def train_and_analyze(X, y, model_name=""):
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train LightGBM model
    model = lgb.LGBMRegressor(random_state=42)
    model.fit(X_train_scaled, y_train)
    
    # Calculate SHAP values
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test_scaled)
    
    # Create SHAP summary plot
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_test, feature_names=features, 
                     show=False, plot_size=(10, 8))
    plt.title(f'SHAP Summary Plot for {model_name} Formulations')
    plt.tight_layout()
    plt.savefig(f'shap_summary_{model_name.lower()}.png')
    plt.close()
    
    # Calculate and return feature importance
    feature_importance = pd.DataFrame({
        'Feature': features,
        'Importance': np.abs(shap_values).mean(0)
    })
    return feature_importance.sort_values('Importance', ascending=False)

# Analyze PLGA formulations
plga_importance = train_and_analyze(X[plga_mask], y[plga_mask], "PLGA")

# Analyze non-PLGA formulations
non_plga_importance = train_and_analyze(X[non_plga_mask], y[non_plga_mask], "Non-PLGA")

# Save feature importance results
plga_importance.to_csv('plga_feature_importance.csv', index=False)
non_plga_importance.to_csv('non_plga_feature_importance.csv', index=False)

# Print comparison
print("\nTop 5 Important Features for PLGA Formulations:")
print(plga_importance.head().to_string(index=False))
print("\nTop 5 Important Features for Non-PLGA Formulations:")
print(non_plga_importance.head().to_string(index=False))

# Create comparison bar plot
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title('Top 5 Features - PLGA')
plt.barh(plga_importance['Feature'][:5][::-1], plga_importance['Importance'][:5][::-1])
plt.subplot(1, 2, 2)
plt.title('Top 5 Features - Non-PLGA')
plt.barh(non_plga_importance['Feature'][:5][::-1], non_plga_importance['Importance'][:5][::-1])
plt.tight_layout()
plt.savefig('feature_importance_comparison.png')
plt.close()