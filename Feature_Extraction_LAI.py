import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.feature_selection import mutual_info_classif
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
import logging
import sys
from datetime import datetime

# Set up logging to both file and console
log_filename = f'feature_extraction_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler(sys.stdout)
    ]
)

def load_dataset():
    """Load and validate the 17-feature dataset"""
    try:
        logging.info("Loading Dataset_17_feat.xlsx...")
        data = pd.read_excel('Dataset_17_feat.xlsx', engine='openpyxl')
        logging.info(f"Dataset loaded successfully with shape: {data.shape}")
        
        # Verify essential columns
        essential_cols = ['LA/GA', 'Polymer_MW', 'CL Ratio', 'Drug_Tm', 'T=1.0']
        missing_cols = [col for col in essential_cols if col not in data.columns]
        if missing_cols:
            logging.warning(f"Missing essential columns: {missing_cols}")
        
        return data
    except Exception as e:
        logging.error(f"Error loading dataset: {str(e)}")
        return None

def process_features(data):
    """Process and enhance features from the dataset"""
    try:
        processed_data = data.copy()
        
        # Identify feature types
        time_cols = ['T=0.25', 'T=0.5', 'T=1.0']
        polymer_cols = ['LA/GA', 'Polymer_MW', 'CL Ratio']
        drug_cols = ['Drug_Tm', 'Drug_Mw', 'Drug_TPSA', 'Drug_LogP']
        
        # Handle missing values using robust methods
        numeric_cols = processed_data.select_dtypes(include=[np.number]).columns
        scaler = RobustScaler()
        
        # Log missing value statistics before processing
        logging.info("Missing value statistics before processing:")
        missing_stats = processed_data[numeric_cols].isna().sum()
        logging.info(f"\n{missing_stats[missing_stats > 0]}")
        
        for col in numeric_cols:
            if col not in time_cols:  # Don't scale time-based features
                mask = processed_data[col].notna()
                if mask.any():
                    # Fill missing values with median before scaling
                    processed_data[col] = processed_data[col].fillna(processed_data[col].median())
                    processed_data.loc[:, col] = scaler.fit_transform(
                        processed_data[col].values.reshape(-1, 1)
                    ).ravel()
        
        # Calculate polymer-specific features
        if set(polymer_cols).issubset(processed_data.columns):
            # LA/GA ratio normalization
            processed_data['LA_Fraction'] = processed_data['LA/GA'].fillna(processed_data['LA/GA'].median()) / \
                                          (1 + processed_data['LA/GA'].fillna(processed_data['LA/GA'].median()))
            
            # Molecular weight features
            processed_data['Log_Polymer_MW'] = np.log1p(processed_data['Polymer_MW'].fillna(processed_data['Polymer_MW'].median()))
            
            # CL Ratio features
            processed_data['CL_Ratio_Normalized'] = processed_data['CL Ratio'].fillna(processed_data['CL Ratio'].median()) / 100.0
        
        # Calculate drug-polymer interaction features
        if 'Drug_Mw' in processed_data.columns and 'Polymer_MW' in processed_data.columns:
            # Fill missing values before calculation
            drug_mw = processed_data['Drug_Mw'].fillna(processed_data['Drug_Mw'].median())
            polymer_mw = processed_data['Polymer_MW'].fillna(processed_data['Polymer_MW'].median())
            processed_data['MW_Ratio'] = drug_mw / polymer_mw
        
        if 'Drug_LogP' in processed_data.columns:
            logp = processed_data['Drug_LogP'].fillna(processed_data['Drug_LogP'].median())
            processed_data['LogP_Normalized'] = (logp - logp.min()) / (logp.max() - logp.min())
        
        # Calculate release profile features
        if all(col in processed_data.columns for col in time_cols):
            processed_data['Initial_Burst'] = processed_data['T=0.25'].fillna(processed_data['T=0.25'].median())
            t1 = processed_data['T=1.0'].fillna(processed_data['T=1.0'].median())
            t025 = processed_data['T=0.25'].fillna(processed_data['T=0.25'].median())
            processed_data['Release_Rate'] = (t1 - t025) / 0.75
        
        # Log missing value statistics after processing
        logging.info("Missing value statistics after processing:")
        final_missing_stats = processed_data.isna().sum()
        logging.info(f"\n{final_missing_stats[final_missing_stats > 0]}")
        
        logging.info(f"Processed features. New shape: {processed_data.shape}")
        return processed_data
        
    except Exception as e:
        logging.error(f"Error processing features: {str(e)}")
        logging.error(f"Full error details:", exc_info=True)
        return None

def analyze_features(data):
    """Analyze feature importance and correlations"""
    try:
        # Select numerical columns excluding time points
        exclude_cols = ['Experimental_index', 'T=0.25', 'T=0.5', 'T=1.0']
        numeric_cols = [col for col in data.select_dtypes(include=[np.number]).columns 
                       if col not in exclude_cols]
        numeric_data = data[numeric_cols]
        
        # Handle NaN values before analysis
        logging.info("Handling missing values in numerical data...")
        numeric_data = numeric_data.fillna(numeric_data.median())
        
        # Create results directory
        os.makedirs('results', exist_ok=True)
        
        # Calculate and plot correlations
        corr_matrix = numeric_data.corr()
        plt.figure(figsize=(15, 12))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, fmt='.2f')
        plt.title('Feature Correlation Matrix')
        plt.tight_layout()
        plt.savefig('results/feature_correlations.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Feature importance analysis
        if 'T=1.0' in data.columns:
            # Define target variable (Slow release if T=1.0 < 0.2)
            y = (data['T=1.0'] < 0.2).astype(int)
            X = numeric_data
            
            # Log feature statistics
            logging.info(f"Feature statistics before MI calculation:")
            logging.info(f"NaN values in features:\n{X.isna().sum()}")
            logging.info(f"Feature ranges:\n{X.describe()}")
            
            # Calculate mutual information scores
            mi_scores = mutual_info_classif(X, y, random_state=42)
            importance_df = pd.DataFrame({
                'Feature': X.columns,
                'Importance': mi_scores
            }).sort_values('Importance', ascending=False)
            
            # Plot feature importance
            plt.figure(figsize=(12, 8))
            sns.barplot(data=importance_df, x='Importance', y='Feature')
            plt.title('Feature Importance for Predicting Slow Release')
            plt.tight_layout()
            plt.savefig('results/feature_importance.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            logging.info("Feature analysis completed successfully")
            return importance_df, corr_matrix
    except Exception as e:
        logging.error(f"Error in feature analysis: {str(e)}")
        logging.error(f"Full error details:", exc_info=True)
        return None, None

def main():
    logging.info("Starting feature extraction and analysis process...")
    
    # Load dataset
    data = load_dataset()
    if data is None:
        return
    
    # Process features
    logging.info("Processing features...")
    processed_data = process_features(data)
    if processed_data is None:
        return
    
    # Analyze features
    logging.info("Analyzing features...")
    importance_df, corr_matrix = analyze_features(processed_data)
    
    if importance_df is not None:
        # Save results
        os.makedirs('results', exist_ok=True)
        
        importance_df.to_csv('results/feature_importance.csv', index=False)
        logging.info("\nTop 5 most important features:")
        print(importance_df.head())
        
        corr_df = pd.DataFrame(corr_matrix)
        corr_df.to_csv('results/feature_correlations.csv')
        
        processed_data.to_csv('results/processed_dataset.csv', index=False)
        logging.info(f"\nProcessed dataset saved with shape: {processed_data.shape}")
        logging.info("Features extracted and analyzed successfully!")
        
        logging.info("\nResults saved in the 'results' directory:")
        for file in ['feature_correlations.png', 'feature_importance.png',
                    'feature_importance.csv', 'feature_correlations.csv',
                    'processed_dataset.csv']:
            logging.info(f"- {file}")

if __name__ == "__main__":
    main()