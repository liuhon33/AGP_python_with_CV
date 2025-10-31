import pandas as pd
import sys

def process_metadata(filepath):
    """
    Loads, filters, and processes the metadata file according to specified rules.
    """
    
    # --- 1. Define Columns to Keep ---
    # Combine all the columns you listed into one list.
    columns_to_select = [
        'sample_name', 'alcohol_consumption', 'appendix_removed', 'cat', 'dog', 
        'lactose', 'multivitamin', 'other_supplement_frequency', 'alcohol_frequency', 
        'artificial_sweeteners', 'cosmetics_frequency', 'exercise_frequency', 
        'fermented_plant_frequency', 'flossing_frequency', 'frozen_dessert_frequency', 
        'fruit_frequency', 'high_fat_red_meat_frequency', 'homecooked_meals_frequency',
        'meat_eggs_frequency', 'milk_cheese_frequency', 'milk_substitute_frequency',
        'olive_oil', 'one_liter_of_water_a_day_frequency', 'poultry_frequency', 
        'prepared_meals_frequency', 'probiotic_frequency', 'ready_to_eat_meals_frequency',
        'red_meat_frequency', 'salted_snacks_frequency',
        'seafood_frequency', 'smoking_frequency', 'sugar_sweetened_drink_frequency',
        'sugary_sweets_frequency', 'teethbrushing_frequency', 'vegetable_frequency',
        'vitamin_b_supplement_frequency', 'vitamin_d_supplement_frequency',
        'vivid_dreams', 'whole_eggs', 'whole_grain_frequency',
        'country', 'race', 
        'sex', 'country_of_birth', 'age_corrected', 'bmi', 'weight_kg'
    ]

    # --- 2. Load Data ---
    try:
        # Assuming the file is a CSV. 
        # If it's an Excel file, use: pd.read_excel(filepath)
        # If it's a TSV (tab-separated), use: pd.read_csv(filepath, sep='\t')
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        print(f"Error: The file '{filepath}' was not found.")
        return None
    except Exception as e:
        print(f"Error loading file: {e}")
        return None

    # --- 3. Select Only the Columns You Want ---
    # Check which of the desired columns actually exist in the file
    available_columns = [col for col in columns_to_select if col in df.columns]
    missing_columns = [col for col in columns_to_select if col not in df.columns]
    
    if missing_columns:
        print(f"Warning: The following columns were not found and will be skipped: {missing_columns}")
        
    df = df[available_columns]

    # --- 4. Process 'sex' Column ---
    if 'sex' in df.columns:
        # Normalize to lowercase and remove leading/trailing spaces
        df['sex'] = df['sex'].astype(str).str.lower().str.strip()
        
        # Filter to keep only 'male' and 'female'
        df = df[df['sex'].isin(['male', 'female'])]
        
        # Code 'male' as 0 and 'female' as 1
        df['sex'] = df['sex'].map({'male': 0, 'female': 1})
    else:
        print("Note: 'sex' column not processed as it wasn't in the selected columns.")

    # --- 5. Process 'bmi' Column ---
    if 'bmi' in df.columns:
        # Convert 'bmi' to a number. If any value can't be converted, it becomes NaN (Not a Number).
        df['bmi'] = pd.to_numeric(df['bmi'], errors='coerce')
        
        # Keep only rows where 'bmi' is between 10 and 60 (inclusive).
        # This filter also automatically removes the NaN rows.
        df = df[(df['bmi'] >= 10) & (df['bmi'] <= 60)]
    else:
        print("Note: 'bmi' column not processed as it wasn't in the selected columns.")
        
    return df

# --- Main execution block ---
if __name__ == "__main__":
    
    # --- !!! CHANGE THESE FILENAMES !!! ---
    INPUT_FILE = '/scratch/liuhon33/parallel/AGPMicrobiomeHostPredictions/Data/Cleaned_data/AGP_Metadata.csv'
    OUTPUT_FILE = 'processed_metadata.csv'
    
    print(f"Starting metadata processing for '{INPUT_FILE}'...")
    
    # Run the processing function
    processed_df = process_metadata(INPUT_FILE)
    
    if processed_df is not None:
        print("\n--- Processing Complete ---")
        
        # --- 6. Save the Result ---
        try:
            processed_df.to_csv(OUTPUT_FILE, index=False)
            print(f"\nSuccessfully saved processed data to '{OUTPUT_FILE}'")
            
            print("\nFirst 5 rows of processed data:")
            print(processed_df.head())
            
            print("\nData summary:")
            processed_df.info()
            
        except Exception as e:
            print(f"Error saving file: {e}")