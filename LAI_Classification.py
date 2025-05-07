import pandas as pd
import numpy as np
from sklearn.model_selection import GroupKFold, train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
import seaborn as sns
import traceback
import os

def load_data():
    """Load and prepare the dataset"""
    try:
        data = pd.read_excel('Dataset_17_feat.xlsx', engine='openpyxl')
        print(f"Dataset loaded with shape: {data.shape}")
        
        # Select all relevant features excluding target variables
        exclude_cols = ['T=0.25', 'T=0.5', 'T=1.0']
        features = [col for col in data.columns if col not in exclude_cols]
        
        print("\nUsing features:", features)
        
        X = data[features].copy()
        y = (data['T=1.0'] < 0.2).astype(int)  # 1 for Slow, 0 for Fast/Intermediate
        
        # Create group identifier for cross-validation
        X['Group'] = X.apply(lambda x: f"{x['Drug']}_{x['Polymer']}" if 'Drug' in X.columns and 'Polymer' in X.columns 
                            else f"Group_{x.name}", axis=1)
        groups = X['Group']
        X = X.drop('Group', axis=1)
        
        return X, y, groups, features
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        traceback.print_exc()
        return None, None, None, None

def plot_confusion_matrix(y_true, y_pred, title, output_dir='plots'):
    """Plot and save confusion matrix"""
    os.makedirs(output_dir, exist_ok=True)
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(f'{output_dir}/{title.lower().replace(" ", "_")}.png')
    plt.close()

def plot_precision_recall_curve(y_test, y_proba, fold, output_dir='plots'):
    """Plot and save precision-recall curve"""
    os.makedirs(output_dir, exist_ok=True)
    precision, recall, _ = precision_recall_curve(y_test, y_proba)
    avg_precision = average_precision_score(y_test, y_proba)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, label=f'AP = {avg_precision:.3f}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve - Fold {fold + 1}')
    plt.legend()
    plt.savefig(f'{output_dir}/precision_recall_curve_fold_{fold + 1}.png')
    plt.close()

def main():
    print("Starting LAI classification process...")
    
    # Load and prepare data
    X, y, groups, features = load_data()
    if X is None:
        return
    
    # Handle missing values using robust methods
    imputer = RobustScaler()
    X_scaled = imputer.fit_transform(X.fillna(X.median()))
    
    # Initialize cross-validation
    gkf = GroupKFold(n_splits=5)
    
    # Results storage
    cv_scores = {
        'roc_auc': [], 'precision': [], 'recall': [], 
        'f1': [], 'avg_precision': []
    }
    fold_importances = []
    all_predictions = []
    
    print("\nPerforming group-based cross-validation...")
    
    for fold, (train_idx, val_idx) in enumerate(gkf.split(X_scaled, y, groups)):
        X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        # Apply SMOTE for class balancing
        smote = SMOTE(random_state=42, sampling_strategy='auto')
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
        
        # Initialize model with optimized parameters
        rf_model = RandomForestClassifier(
            n_estimators=200,
            max_depth=8,
            min_samples_split=10,
            min_samples_leaf=4,
            max_features='sqrt',
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )
        
        # Train model
        rf_model.fit(X_train_resampled, y_train_resampled)
        
        # Make predictions
        y_pred = rf_model.predict(X_val)
        y_pred_proba = rf_model.predict_proba(X_val)[:, 1]
        
        # Store predictions
        fold_df = pd.DataFrame({
            'True_Label': y_val,
            'Predicted_Label': y_pred,
            'Probability': y_pred_proba,
            'Fold': fold + 1
        })
        all_predictions.append(fold_df)
        
        # Calculate metrics
        fold_score = roc_auc_score(y_val, y_pred_proba)
        avg_precision = average_precision_score(y_val, y_pred_proba)
        cv_scores['roc_auc'].append(fold_score)
        cv_scores['avg_precision'].append(avg_precision)
        
        # Store feature importance
        fold_importances.append(rf_model.feature_importances_)
        
        # Generate plots
        plot_confusion_matrix(y_val, y_pred, f'Confusion Matrix Fold {fold + 1}')
        plot_precision_recall_curve(y_val, y_pred_proba, fold)
        
        print(f"\nFold {fold + 1} Results:")
        print(f"ROC-AUC Score: {fold_score:.4f}")
        print(f"Average Precision Score: {avg_precision:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_val, y_pred))
    
    # Calculate and display overall results
    print("\nOverall Cross-validation Results:")
    for metric, scores in cv_scores.items():
        if scores:
            print(f"Mean {metric}: {np.mean(scores):.4f} (+/- {np.std(scores) * 2:.4f})")
    
    # Average feature importance
    mean_importance = np.mean(fold_importances, axis=0)
    importance_df = pd.DataFrame({
        'Feature': features,
        'Importance': mean_importance
    }).sort_values('Importance', ascending=False)
    
    # Save results
    print("\nSaving results...")
    os.makedirs('results', exist_ok=True)
    
    # Save feature importance
    importance_df.to_csv('results/lai_feature_importance.csv', index=False)
    
    # Save all predictions
    predictions_df = pd.concat(all_predictions)
    predictions_df.to_csv('results/lai_predictions.csv', index=False)
    
    # Save detailed metrics
    metrics_df = pd.DataFrame(cv_scores)
    metrics_df.to_csv('results/lai_metrics.csv', index=False)
    
    # Plot overall feature importance
    plt.figure(figsize=(12, 6))
    importance_df.plot(kind='bar', x='Feature', y='Importance')
    plt.title('Feature Importance in LAI Classification')
    plt.xlabel('Features')
    plt.ylabel('Importance Score')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('results/lai_feature_importance.png')
    plt.close()
    
    print("\nClassification process completed successfully!")
    print("Results and visualizations have been saved to the 'results' directory.")

if __name__ == "__main__":
    main()