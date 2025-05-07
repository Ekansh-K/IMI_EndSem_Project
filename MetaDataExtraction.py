import os
import re
import pandas as pd
from pdfplumber import open as pdf_open
import PyPDF2


# ---------------------------
# Step 1: Helper Functions
# ---------------------------

def read_pdf_text(pdf_path):
    """Extract raw text from a PDF."""
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ''
        for page in reader.pages:
            text += page.extract_text()
        return text


def extract_formulation_data(text):
    """Extract key formulation parameters using regex."""
    dp_group_match = re.search(r'(?:[A-Z][a-z]+-[A-Z]{3})', text)  # e.g., DEX-PLGA
    la_ga_match = re.search(r'LA/GA\s*[:=]?\s*(\d+\.?\d*)', text)
    mw_match = re.search(r'Polymer MW\s*[:=]?\s*(\d+)', text)
    dlc_match = re.search(r'DLC\s*[:=]?\s*(\d+\.?\d*)', text)

    return {
        "DP_Group": dp_group_match.group(0) if dp_group_match else None,
        "LA/GA": la_ga_match.group(1) if la_ga_match else None,
        "Polymer_MW": mw_match.group(1) if mw_match else None,
        "DLC": dlc_match.group(1) if dlc_match else None
    }


def extract_release_profile(pdf_path):
    """Try to extract tabular release data using pdfplumber."""
    try:
        with pdf_open(pdf_path) as pdf:
            all_tables = []
            for page in pdf.pages:
                for table in page.extract_tables():
                    all_tables.append(table)
            return all_tables
    except Exception as e:
        print(f"Error extracting tables from {pdf_path}: {e}")
        return []


def process_paper(pdf_path):
    """Process one paper: extract metadata + release data."""
    text = read_pdf_text(pdf_path)
    base_data = extract_formulation_data(text)
    tables = extract_release_profile(pdf_path)

    rows = []
    for table in tables:
        headers = table[0]
        if any("Time" in str(h).lower() or "Day" in str(h).lower() for h in headers):
            for row in table[1:]:
                if len(row) >= 2:
                    time_val = row[0].strip() if isinstance(row[0], str) else str(row[0])
                    release_val = row[-1].strip() if isinstance(row[-1], str) else str(row[-1])

                    if time_val.replace('.', '', 1).isdigit() and release_val.replace('.', '', 1).isdigit():
                        rows.append({
                            **base_data,
                            "Time": float(time_val),
                            "Release": float(release_val)
                        })
    return rows


# ---------------------------
# Step 2: Process All Papers in Folder
# ---------------------------

input_folder = "Potential_Papers"
all_extracted_rows = []

for filename in os.listdir(input_folder):
    if filename.lower().endswith(".pdf"):
        path = os.path.join(input_folder, filename)
        print(f"Processing {filename}...")
        extracted_data = process_paper(path)
        all_extracted_rows.extend(extracted_data)

# Convert to DataFrame
new_data_df = pd.DataFrame(all_extracted_rows)


# ---------------------------
# Step 3: Load Original Dataset
# ---------------------------

original_df = pd.read_excel("Dataset_17_feat.xlsx")


# ---------------------------
# Step 4: Align Columns Before Merging
# ---------------------------

# Define relevant columns
relevant_columns = [
    "DP_Group", "LA/GA", "Polymer_MW", "DLC", "Time", "Release"
]

# Filter both datasets
original_filtered = original_df[relevant_columns].copy()
new_filtered = new_data_df[relevant_columns].dropna(subset=["DP_Group", "Time", "Release"])

# Add source identifier
original_filtered["Source"] = "Original"
new_filtered["Source"] = "Extracted"


# ---------------------------
# Step 5: Combine & Save Final Dataset
# ---------------------------

combined_df = pd.concat([original_filtered, new_filtered], ignore_index=True)
combined_df.to_excel("merged_dataset.xlsx", index=False)

print("✅ Merged dataset saved to 'merged_dataset.xlsx'")