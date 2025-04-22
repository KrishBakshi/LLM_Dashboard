
import pandas as pd

def clean_column_name(col):
    col = str(col).strip().lstrip('- ').strip()
    return col

def preprocess_excel(raw_excel_path, output_csv_path):
    # Load with both header rows
    df_raw = pd.read_excel(raw_excel_path, sheet_name="Raw Data", header=[0, 1])

    # Merge headers and clean
    cleaned_cols = []
    for col1, col2 in df_raw.columns:
        col1 = clean_column_name(col1)
        col2 = clean_column_name(col2)
        merged = f"{col1} - {col2}" if col2 else col1
        cleaned_cols.append(merged.strip())

    df_raw.columns = [clean_column_name(col) for col in cleaned_cols]

    # Drop first two rows which often contain repeated labels or NaNs
    df_raw = df_raw.drop(index=[0, 1]).reset_index(drop=True)

    # Replace encoding junk (like â€“) with proper hyphens
    df_raw = df_raw.applymap(
        lambda x: x.replace('â€“', '-').replace('â€œ', '"').replace('â€', '"') if isinstance(x, str) else x
    )

    # Save cleaned data
    df_raw.to_csv(output_csv_path, index=False, encoding='utf-8-sig')
    print(f"✅ Cleaned data saved to: {output_csv_path}")

# Run it
if __name__ == "__main__":
    preprocess_excel("Data Work_H.xlsx", "preprocessed_data.csv")
