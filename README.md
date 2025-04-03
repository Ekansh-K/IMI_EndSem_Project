-Key Features to be implemented 
1. GCNN with Transfer Learning 
What It Is : 
• GCNN processes molecular graphs to extract structural and physicochemical 
information about drugs and polymers. 
• Transfer Learning : Pre-train GCNN on large chemical databases (e.g., PubChem) 
and fine-tune it on the PLAI dataset. 
Why It’s Beneficial : 
• Captures rich molecular structural information (e.g., functional groups, bond types). 
• Generalizes to unseen molecules by leveraging pre-trained knowledge. 
• Improves performance in low-data scenarios (e.g., the PLAI dataset with 181 drug 
release profiles). 
Implementation Steps : 
1. Reconstruct Molecular Structures : 
• Use RDKit to generate SMILES strings for drugs and polymers in the PLAI 
dataset. 
2. Pre-train GCNN : 
• Train GCNN on large molecular databases (e.g., ZINC, ChEMBL) using graph
based representations. Or we can start with a the use of preexisting pre 
trained model  
3. Fine-tune GCNN : 
• Freeze most GCNN layers and fine-tune the last few layers on the PLAI 
dataset. 
4. Concatenate Features : 
• Combine GCNN-generated fingerprint vectors with traditional descriptors to 
form the Intermediate Input Vector (IIV) . 
5. Train Models : 
• Use IIVs as inputs for models like LGBM, SNN, or neural networks. 
2. Siamese Neural Networks (SNN) ( Allows for Classification) 
What It Is : 
• SNN compares pairs of inputs (e.g., successful vs. unsuccessful formulations) to 
classify new examples based on similarity. 
Why It’s Beneficial : 
• Excels in low-data regimes (e.g., 181 drug release profiles). 
• Learns similarity metrics for one-shot/few-shot learning. 
• Provides probabilistic outputs for success/failure classification. 
Implementation Steps : 
1. Define Success Criteria : 
• Set thresholds for fractional drug release (e.g., T=1.0 > 0.8 for slow-release). 
2. Prepare IIV : 
• Concatenate GCNN fingerprint vectors with traditional descriptors (e.g., 
DLC, SA-V, Time). 
3. Train SNN : 
• Use pairs of labeled examples (same-class and different-class pairs). 
• Optimize contrastive loss to minimize distance between similar pairs and 
maximize it for dissimilar pairs. 
4. Predict New Formulations : 
• Compare new formulations to a support set of known successful examples. 
• Aggregate similarity scores to classify the new formulation. 
• SNN would enable classification of formulations as "successful" or "unsuccessful." 
3. Active Learning 
What It Is : 
• Iteratively select the most informative samples for experimental validation. 
Why It’s Beneficial : 
• Maximizes the value of new experiments. 
• Reduces the cost of data collection. 
Implementation Steps : 
1. Identify Uncertain Predictions : 
• Use model uncertainty (e.g., low confidence scores) to prioritize 
experiments. 
2. Update Dataset : 
• Add newly validated samples to the training set. 
3. Retrain Model : 
• Continuously improve the model with active learning loops. -Sub Features that need to be worked : 
1. Dataset Expansion via Web Scraping 
What It Is : 
• Expand the PLAI dataset by scraping recent studies (past 3 years) to include new 
drug-polymer combinations and experimental conditions. 
Why It’s Beneficial : 
• Increases dataset size and diversity, improving model robustness. 
• Reduces bias and overfitting in low-data scenarios. 
Implementation Steps : 
1. Scrape Data : 
• Use tools like BeautifulSoup or Scrapy to extract data from recent 
publications. 
2. Extract Numerical Data : 
• Use GetData Graph Digitizer to extract fractional drug release values from 
f
 igures. 
3. Clean and Standardize : 
• Remove duplicates, resolve inconsistencies, and normalize features. 
4. Merge with Existing Dataset : 
• Combine new data with the original PLAI dataset for training. 
2. Addressing Noise with Clustering using new algorithm on the New scrapped dataset 
What It Is : 
• Use clustering algorithms (e.g., DBSCAN , Hierarchical Clustering ) to identify and 
remove noisy or outlier data points. 
Why It’s Beneficial : 
• Reduces noise that could distort PCA or model training. 
• Improves interpretability and generalization. 
Implementation Steps : 
1. Apply Clustering : 
• Cluster the dataset using DBSCAN or hierarchical clustering. 
2. Identify Outliers : 
• Flag data points in low-density clusters or labeled as noise. 
3. Remove or Impute : 
• Remove outliers or impute values using techniques like KNN imputation. 
3.Implement k fold Cross Validation similar to the paper  
• Use group-based cross-validation to ensure splits respect drug-polymer 
combinations. 
Why It’s Beneficial : 
• Prevents data leakage and ensures realistic generalization. 
Implementation Steps : 
1. Group by Drug-Polymer Pairs : 
• Split data such that no drug-polymer combination appears in both training 
and test sets. 
2. Evaluate Models : 
• Use nested cross-validation for hyperparameter tuning. 
4. Handling Class Imbalance(Due to the implementation of Snn ) 
Using Balanced Random Forest or RUSBoost or SMOTE 
What It Is : 
• Address class imbalance (e.g., more "unsuccessful" formulations than "successful" 
ones) using techniques like SMOTE or class weighting . 
Without addressing this imbalance, the model might: 
• Predict the majority class most of the time, leading to poor performance on the 
minority class. 
• Fail to capture critical patterns that differentiate successful formulations. 
Why It’s Beneficial : 
• Prevents models from being biased toward the majority class. 
• Improves performance on minority classes (e.g., rare successful formulations). 
Implementation Steps : 
1. Analyze Class Distribution : 
• Use value_counts() to check the balance of success/failure labels. 
2. Apply SMOTE : 
• Generate synthetic samples for the minority class. 
3. Class Weighting : 
• Assign higher weights to minority classes during model training. 
5. Train and test Different Models like LGBM and also do shapley analysis from the 
result of the model’s prediction from new dataset  
Implementation Steps : 
1. Train Model : 
• Train LGBM, SNN, or GCNN+SNN on the dataset. 
2. Compute SHAP Values : 
• Use shap.TreeExplainer for tree-based models or shap.GradientExplainer for 
neural networks. 
3. Visualize : 
• Generate summary plots, force plots, or heatmaps to interpret results. - Experimental Implementations(Require a Lot of Domain 
Knowledge and Understanding Maths and probability) : 
1. Incoperating Physic Model 
What It Is : 
• Integrate domain knowledge (e.g., polymer degradation kinetics, diffusion 
equations) into ML models. 
Why It’s Beneficial : 
• Ensures predictions align with physical laws. 
• Reduces overfitting to noise. 
Implementation Steps : 
1. Define Physical Constraints : 
• Incorporate equations like Fick’s Law or hydrolysis rates into the loss 
function. 
2. Hybrid Models : 
• Combine ML models (e.g., LGBM, SNN) with physics-based simulations. 
2.Bayesian Optimization for Hyperparameter Tuning 
What It Is : 
• Use Bayesian optimization (e.g., scikit-optimize) to tune hyperparameters 
efficiently. 
Why It’s Beneficial : 
• Finds optimal hyperparameters faster than grid/random search. 
• Works well for expensive-to-train models like GCNN+SNN. 
Implementation Steps : 
1. Define Search Space : 
• Specify hyperparameters (e.g., learning rate, number of GCNN layers). 
2. Optimize : 
• Run Bayesian optimization to minimize validation loss (e.g., MAE). 
PLAI Paper Context : 
• The original study used random grid search. Bayesian optimization would improve 
efficiency. 
3.Synthetic Data Generation & Model Implementation  
What It Is : 
• Generate synthetic data using the trained SNN or GCNN+SNN to augment the 
dataset. 
Why It’s Beneficial : 
• Addresses data scarcity by creating realistic synthetic examples. 
• Enables training of complex models like neural networks.

Implementation Steps : 
1. Generate Synthetic Examples : 
• Use the trained SNN to sample new conditions (e.g., pH, temperature). 
2. Label Synthetically : 
• Assign labels based on the model’s predictions. 
3. Train on Expanded Dataset : 
• Combine synthetic and real data for training. 
4.  Train Decision Tree : 
• Fit a decision tree on the synthetic data. 
5. Extract Rules : 
• Visualize the decision tree to identify critical thresholds 
