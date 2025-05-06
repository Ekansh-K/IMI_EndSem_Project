import pandas as pd
import pubchempy as pcp
from time import sleep
from tqdm import tqdm
from rdkit import Chem

# === 1. Load dataset ===
df = pd.read_excel('Dataset_17_feat.xlsx')
df['Drug'] = df['DP_Group'].str.split('-').str[0]
unique_drugs = df['Drug'].unique().tolist()

# === 2. Function to fetch SMILES ===
def get_smiles(drug_name, delay=1):
    try:
        compounds = pcp.get_compounds(drug_name, 'name')
        if compounds:
            sleep(delay)
            return compounds[0].canonical_smiles
        else:
            print(f"❌ No compound found for '{drug_name}'")
            return None
    except Exception as e:
        print(f" Error fetching SMILES for '{drug_name}': {e}")
        return None

# === 3. Get SMILES for all unique drugs ===
drug_smiles_map = {}
for drug in tqdm(unique_drugs, desc="Fetching SMILES"):
    smiles = get_smiles(drug)
    drug_smiles_map[drug] = smiles

# === 4. Save to CSV ===
smiles_df = pd.DataFrame.from_dict(drug_smiles_map, orient='index', columns=['SMILES'])
smiles_df.index.name = 'Drug'
smiles_df.reset_index(inplace=True)
smiles_df.to_csv('drug_smiles.csv', index=False)
print("\n✅ SMILES saved to 'drug_smiles.csv'")

# === 5. Merge with original dataset ===
df_with_smiles = pd.merge(df, smiles_df, left_on='Drug', right_on='Drug', how='left')

# === 6. Validate SMILES ===
def is_valid_smiles(smiles):
    if pd.isna(smiles):
        return False
    mol = Chem.MolFromSmiles(smiles)
    return mol is not None

df_with_smiles['Valid_SMILES'] = df_with_smiles['SMILES'].apply(is_valid_smiles)

# Save final dataset
df_with_smiles.to_excel('Dataset_with_SMILES.xlsx', index=False)
print("✅ Final dataset saved to 'Dataset_with_SMILES.xlsx'")