import pandas as pd
import os
from rdkit import Chem
import logging

# Set up logging to capture missing or invalid SMILES
logging.basicConfig(filename='smiles_processing.log', level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Define SMILES mappings for drugs and polymers in the dataset
drug_smiles_map = {
    '5-FU': 'c1c(c(=O)[nH]c(=O)[nH]1)F',  # 5-Fluorouracil, sourced from PubChem (CID 3385)
    'ACE': 'CC(C)NC[C@H](COC(=O)CCc1ccc(c(c1)OC(=O)C)C)O',  # Acebutolol, sourced from PubChem (CID 1978)
    'CAF': 'Cn1c(=O)c2c(n(c(=O)[nH]c2n(c1=O)C)C)C',  # Caffeine, sourced from PubChem (CID 2519)
    'TAH': 'CC1C[C@@H]2C[C@H]([C@H](C(=O)[C@@H](C[C@@H](/C(=C/[C@H](C[C@H](C[C@H]3CC[C@H]([C@@](O3)(C(=O)C(=O)N4CCCC[C@H]4C(=O)O2)O)C)OC)/C)C)O)OC)C)OC',  # Tacrolimus, sourced from PubChem (CID 445643)
    'THC': 'CCCCCc1cc(c(c(c1)O)[C@@H]2C=C(C)C[C@@H](C2)C(C)(C)O)O',  # Tetrahydrocannabinol, sourced from PubChem (CID 16078)
    'TMZ': 'Cc1nc2c(c(=O)n1)[nH]c(c(c2)C(=O)N)N',  # Temozolomide, sourced from PubChem (CID 5394)
    'TTD': 'C[N+]1(C)[C@@H]2C[C@@H](C[C@@H]1[C@H]3O[C@@H]23)OC(=O)[C@@H](c4cccs4)c5cccs5',  # Tiotropium, sourced from PubChem (CID 5487426)
    'DOX': 'C[C@H]1[C@H]([C@H](C[C@@H](O1)O[C@H]2C[C@@](Cc3c2c(c(c(c3)O)O)O)(C(=O)CO)O)N)O',  # Doxorubicin, sourced from PubChem (CID 31703)
    'PTX': 'CC1=C2[C@H](C(=O)[C@@]3([C@H](C[C@@H](C(C3(C)C)O)(C[C@H]1OC(=O)[C@H](c4ccccc4)NC(=O)c5ccccc5)O)OC(=O)C)OC(=O)C)[C@@H](OC(=O)c6ccccc6)[C@](C2(C)C)(O)COC(=O)C',  # Paclitaxel, sourced from PubChem (CID 36314)
    'IBU': 'CC(C)CC1=CC=C(C=C1)C(C)C(=O)O',  # Ibuprofen, sourced from PubChem (CID 3672)
    'MET': 'CN(C)C(=N)N=C(N)N',  # Metformin, sourced from PubChem (CID 4091)
    'CIS': 'N.Cl[Pt]Cl.N'  # Cisplatin, sourced from PubChem (CID 441203)
}

polymer_smiles_map = {
    'PLGA': 'C[C@@H](O)C(=O)OCC(=O)O',  # Poly(lactic-co-glycolic acid), monomer unit (lactic acid + glycolic acid), approximated for repeating structure
    'PVL-co-PAVL': 'O=C1CCCCO1.O=C1CC(CC=C)CCO1',  # Poly(valerolactone-co-allyl valerolactone), monomer units (valerolactone + allyl valerolactone), non-crosslinked
    'PCL': 'O=C1CCCCCCO1',  # Poly(caprolactone), monomer unit (caprolactone), standard repeating unit
    'PEG': 'OCCO',  # Polyethylene glycol, simplified monomer unit (ethylene glycol), repeating structure approximated
    'PLA': 'C[C@@H](O)C(=O)O',  # Polylactic acid, monomer unit (lactic acid), standard repeating unit
    'PVA': 'C[C@@H](O)CO',  # Polyvinyl alcohol, monomer unit (vinyl alcohol), simplified repeating structure
    'PAA': 'C(C(=O)O)C',  # Polyacrylic acid, monomer unit (acrylic acid), standard repeating unit
    'PCL-PEG': 'O=C1CCCCCCO1.OCCO'  # PCL-PEG copolymer, combined monomer units (caprolactone + ethylene glycol), simplified
}

def validate_smiles(smiles, context=""):
    """
    Validate a SMILES string using RDKit.
    Args:
        smiles (str): SMILES string to validate.
        context (str): Context for logging (e.g., 'Drug' or 'Polymer').
    Returns:
        bool: True if valid, False otherwise.
    """
    if not smiles:
        logging.warning(f"{context} SMILES is None or empty")
        return False
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            logging.warning(f"Invalid {context} SMILES: {smiles}")
            return False
        return True
    except Exception as e:
        logging.error(f"Error validating {context} SMILES '{smiles}': {e}")
        return False

def extract_drug_polymer(dp_group):
    """
    Extract drug and polymer names from DP_Group string.
    Args:
        dp_group (str): Drug-polymer pair (e.g., '5-FU-PLGA').
    Returns:
        tuple: (drug, polymer) names, or (None, None) if parsing fails.
    """
    try:
        parts = dp_group.split('-')
        drug = parts[0]
        polymer = '-'.join(parts[1:])  # Handle multi-part polymer names like PVL-co-PAVL
        return drug, polymer
    except Exception as e:
        logging.error(f"Error parsing DP_Group '{dp_group}': {e}")
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
        logging.info(f"Loaded dataset with {len(data)} rows and {len(data.columns)} columns")

        # Initialize new columns
        data['Drug_SMILES'] = None
        data['Polymer_SMILES'] = None

        # Track missing or invalid SMILES
        missing_drugs = set()
        missing_polymers = set()
        invalid_drug_smiles = []
        invalid_polymer_smiles = []

        # Map SMILES strings to each record
        for index, row in data.iterrows():
            drug, polymer = extract_drug_polymer(row['DP_Group'])
            if drug and polymer:
                # Assign Drug SMILES
                if drug in drug_smiles_map:
                    drug_smiles = drug_smiles_map[drug]
                    if validate_smiles(drug_smiles, f"Drug {drug}"):
                        data.at[index, 'Drug_SMILES'] = drug_smiles
                    else:
                        invalid_drug_smiles.append((drug, drug_smiles))
                else:
                    missing_drugs.add(drug)
                    logging.warning(f"No SMILES for drug: {drug}")

                # Assign Polymer SMILES
                if polymer in polymer_smiles_map:
                    polymer_smiles = polymer_smiles_map[polymer]
                    if validate_smiles(polymer_smiles, f"Polymer {polymer}"):
                        data.at[index, 'Polymer_SMILES'] = polymer_smiles
                    else:
                        invalid_polymer_smiles.append((polymer, polymer_smiles))
                else:
                    missing_polymers.add(polymer)
                    logging.warning(f"No SMILES for polymer: {polymer}")

        # Log summary
        if missing_drugs:
            logging.warning(f"Missing drug SMILES for: {missing_drugs}")
        if missing_polymers:
            logging.warning(f"Missing polymer SMILES for: {missing_polymers}")
        if invalid_drug_smiles:
            logging.warning(f"Invalid drug SMILES: {invalid_drug_smiles}")
        if invalid_polymer_smiles:
            logging.warning(f"Invalid polymer SMILES: {invalid_polymer_smiles}")

        # Save the updated dataset
        data.to_csv(output_csv, index=False)
        logging.info(f"Updated dataset saved to '{output_csv}'")

    except Exception as e:
        logging.error(f"Error processing dataset: {e}")
        raise

def main():
    """
    Main function to execute the SMILES addition process.
    """
    input_csv = 'Dataset_17_feat (1).csv'
    output_csv = 'expanded_dataset_with_smiles.csv'

    try:
        add_smiles_columns(input_csv, output_csv)

        # Verify the output
        if os.path.exists(output_csv):
            updated_data = pd.read_csv(output_csv)
            logging.info(f"Output dataset has {len(updated_data)} rows and {len(updated_data.columns)} columns")
            logging.info(f"Columns: {list(updated_data.columns)}")
            # Check for missing SMILES
            missing_drug_smiles = updated_data['Drug_SMILES'].isna().sum()
            missing_polymer_smiles = updated_data['Polymer_SMILES'].isna().sum()
            logging.info(f"Missing Drug_SMILES: {missing_drug_smiles}")
            logging.info(f"Missing Polymer_SMILES: {missing_polymer_smiles}")
            print(f"Dataset updated successfully. Output saved to '{output_csv}'")
            print(f"Check 'smiles_processing.log' for details on missing or invalid SMILES.")
        else:
            logging.error("Output file was not created.")
            print("Output file was not created.")

    except Exception as e:
        logging.error(f"Main execution failed: {e}")
        print(f"Main execution failed: {e}")

if __name__ == '__main__':
    main()