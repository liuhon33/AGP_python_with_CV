#!/usr/bin/env python3
import pandas as pd


def process_metadata(filepath):
    """
    Loads, filters, and processes the metadata file according to specified rules.

    - Select specific columns (skip missing with warning)
    - sex: keep only 'male' and 'female' exactly; code male=0, female=1
    - bmi: numeric; keep 10..60 inclusive
    - bowel_movement_frequency: exact mapping from provided categories -> 0..5
        * 'Not provided'/'Unspecified' -> impute with rounded mean of valid codes
    - sleep_duration: exact mapping from provided categories -> integers
        * 'Not provided'/'Unspecified' -> impute with rounded mean of valid codes
    """

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

        # NEW:
        'bowel_movement_frequency',
        'sleep_duration',

        'country', 'race',
        'sex', 'country_of_birth', 'age_corrected', 'bmi', 'weight_kg'
    ]

    # --- Load ---
    try:
        df = pd.read_csv(filepath)  # change to TSV if needed
    except FileNotFoundError:
        print(f"Error: The file '{filepath}' was not found.")
        return None
    except Exception as e:
        print(f"Error loading file: {e}")
        return None

    # --- Select columns ---
    available_columns = [c for c in columns_to_select if c in df.columns]
    missing_columns = [c for c in columns_to_select if c not in df.columns]
    if missing_columns:
        print(f"Warning: The following columns were not found and will be skipped: {missing_columns}")

    df = df[available_columns].copy()

    # --- sex (EXACT match only) ---
    if 'sex' in df.columns:
        df = df[df['sex'].isin(['male', 'female'])].copy()
        df['sex'] = df['sex'].map({'male': 0, 'female': 1}).astype('Int64')

    # --- bmi ---
    if 'bmi' in df.columns:
        df['bmi'] = pd.to_numeric(df['bmi'], errors='coerce')
        df = df[(df['bmi'] >= 10) & (df['bmi'] <= 60)].copy()

    # --- bowel_movement_frequency (EXACT mapping) ---
    if 'bowel_movement_frequency' in df.columns:
        bowel_map = {
            'Less than one': 0,
            'One': 1,
            'Two': 2,
            'Three': 3,
            'Four': 4,
            'Five or more': 5,

            # treat these as missing then impute:
            'Not provided': pd.NA,
            'Unspecified': pd.NA
        }

        bowel_num = df['bowel_movement_frequency'].map(bowel_map).astype('Float64')  # float for mean
        mean_val = bowel_num.mean(skipna=True)

        # impute only when original was exactly Not provided / Unspecified
        to_impute = df['bowel_movement_frequency'].isin(['Not provided', 'Unspecified'])
        if pd.notna(mean_val):
            bowel_num = bowel_num.mask(to_impute, int(round(float(mean_val))))

        df['bowel_movement_frequency'] = bowel_num.astype('Int64')

    # --- sleep_duration (EXACT mapping) ---
    if 'sleep_duration' in df.columns:
        sleep_map = {
            'Less than 5 hours': 3,
            '5-6 hours': 6,
            '6-7 hours': 7,
            '7-8 hours': 8,
            '8 or more hours': 10,

            # treat these as missing then impute:
            'Not provided': pd.NA,
            'Unspecified': pd.NA
        }

        sleep_num = df['sleep_duration'].map(sleep_map).astype('Float64')
        mean_val = sleep_num.mean(skipna=True)

        to_impute = df['sleep_duration'].isin(['Not provided', 'Unspecified'])
        if pd.notna(mean_val):
            sleep_num = sleep_num.mask(to_impute, int(round(float(mean_val))))

        df['sleep_duration'] = sleep_num.astype('Int64')

    return df


if __name__ == "__main__":
    INPUT_FILE = '/scratch/liuhon33/parallel/AGPMicrobiomeHostPredictions/Data/Cleaned_data/AGP_Metadata.csv'
    OUTPUT_FILE = 'processed_metadata.csv'
    print(f"Starting metadata processing for '{INPUT_FILE}'...")

    processed_df = process_metadata(INPUT_FILE)

    if processed_df is not None:
        processed_df.to_csv(OUTPUT_FILE, index=False)
        print(f"\nSaved to: {OUTPUT_FILE}")

        print("\nData summary:")
        processed_df.info()

        if 'bowel_movement_frequency' in processed_df.columns:
            print("\nBowel movement frequency (coded) value counts:")
            print(processed_df['bowel_movement_frequency'].value_counts(dropna=False).sort_index())

        if 'sleep_duration' in processed_df.columns:
            print("\nSleep duration (coded) value counts:")
            print(processed_df['sleep_duration'].value_counts(dropna=False).sort_index())