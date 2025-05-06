import pandas as pd
import uuid
import os

# Define SMILES mappings for drugs and polymers
drug_smiles_map = {
    '5-FU': 'c1c(c(=O)[nH]c(=O)[nH]1)F',  # 5-Fluorouracil
    'ACE': 'CC(C)NC[C@H](COC(=O)CCc1ccc(c(c1)OC(=O)C)C)O',  # Acebutolol
    'CAF': 'Cn1c(=O)c2c(n(c(=O)[nH]c2n(c1=O)C)C)C',  # Caffeine
    'TAH': 'CC1C[C@@H]2C[C@H]([C@H](C(=O)[C@@H](C[C@@H](/C(=C/[C@H](C[C@H](C[C@H]3CC[C@H]([C@@](O3)(C(=O)C(=O)N4CCCC[C@H]4C(=O)O2)O)C)OC)/C)C)O)OC)C)OC',  # Tacrolimus
    'THC': 'CCCCCc1cc(c(c(c1)O)[C@@H]2C=C(C)C[C@@H](C2)C(C)(C)O)O',  # Tetrahydrocannabinol
    'TMZ': 'Cc1nc2c(c(=O)n1)[nH]c(c(c2)C(=O)N)N',  # Temozolomide
    'TTD': 'C[N+]1(C)[C@@H]2C[C@@H](C[C@@H]1[C@H]3O[C@@H]23)OC(=O)[C@@H](c4cccs4)c5cccs5'  # Tiotropium
}

polymer_smiles_map = {
    'PLGA': 'C[C@@H](O)C(=O)OCC(=O)O',  # Poly(lactic-co-glycolic acid) monomer unit
    'PVL-co-PAVL': 'O=C1CCCCO1.O=C1CC(CC=C)CCO1',  # Poly(valerolactone-co-allyl valerolactone) monomer units
    'PCL': 'O=C1CCCCCCO1'  # Poly(caprolactone) monomer unit
}

def extract_drug_polymer(dp_group):
    """
    Extract drug and polymer names from DP_Group string.
    Args:
        dp_group (str): Drug-polymer pair (e.g., '5-FU-PLGA').
    Returns:
        tuple: (drug, polymer) names.
    """
    try:
        dp_group = str(dp_group)  # Convert to string to handle non-string values
        parts = dp_group.split('-')
        drug = parts[0]
        polymer = '-'.join(parts[1:])  # Handle multi-part polymer names like PVL-co-PAVL
        return drug, polymer
    except Exception as e:
        print(f"Error parsing DP_Group '{dp_group}': {e}")
        return None, None

def add_smiles_columns(input_csv, output_csv):
    """
    Add Drug_SMILES and Polymer_SMILES columns to the dataset.
    Args:
        input_csv (str): Path to input CSV file.
        output_csv (str): Path to output CSV file.
    Returns:
        None
    """
    try:
        # Load the dataset
        if not os.path.exists(input_csv):
            raise FileNotFoundError(f"Input file '{input_csv}' not found.")
        data = pd.read_csv(input_csv)

        # Initialize new columns
        data['Drug_SMILES'] = None
        data['Polymer_SMILES'] = None

        # Map SMILES strings to each record
        missing_drugs = set()
        missing_polymers = set()
        for index, row in data.iterrows():
            drug, polymer = extract_drug_polymer(row['DP_Group'])
            if drug and polymer:
                # Assign Drug SMILES
                if drug in drug_smiles_map:
                    data.at[index, 'Drug_SMILES'] = drug_smiles_map[drug]
                else:
                    missing_drugs.add(drug)
                
                # Assign Polymer SMILES
                if polymer in polymer_smiles_map:
                    data.at[index, 'Polymer_SMILES'] = polymer_smiles_map[polymer]
                else:
                    missing_polymers.add(polymer)

        # Warn about missing SMILES
        if missing_drugs:
            print(f"Warning: No SMILES found for drugs: {missing_drugs}")
        if missing_polymers:
            print(f"Warning: No SMILES found for polymers: {missing_polymers}")

        # Save the updated dataset
        data.to_csv(output_csv, index=False)
        print(f"Updated dataset saved to '{output_csv}'")

    except Exception as e:
        print(f"Error processing dataset: {e}")
        raise

def main():
    """
    Main function to execute the SMILES addition process.
    """
    input_csv = 'Dataset_17_feat.csv'  # Fixed filename format
    output_csv = 'expanded_dataset_with_smiles.csv'
    
    try:
        add_smiles_columns(input_csv, output_csv)
        
        # Verify the output
        if os.path.exists(output_csv):
            updated_data = pd.read_csv(output_csv)
            print(f"Output dataset has {len(updated_data)} rows and {len(updated_data.columns)} columns")
            print(f"Columns: {list(updated_data.columns)}")
            # Check for missing SMILES
            missing_drug_smiles = updated_data['Drug_SMILES'].isna().sum()
            missing_polymer_smiles = updated_data['Polymer_SMILES'].isna().sum()
            print(f"Missing Drug_SMILES: {missing_drug_smiles}")
            print(f"Missing Polymer_SMILES: {missing_polymer_smiles}")
        else:
            print("Output file was not created.")
            
    except Exception as e:
        print(f"Main execution failed: {e}")

if __name__ == '__main__':
    main()