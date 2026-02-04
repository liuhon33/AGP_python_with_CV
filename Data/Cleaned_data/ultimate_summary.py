import pandas as pd
import numpy as np
import os
import warnings

def generate_column_stats(input_path, id_col_name):
    # 1. Read Data
    ext = os.path.splitext(input_path)[1].lower()
    
    # Using 'on_bad_lines' skip is roughly similar to name_repair/loose reading, 
    # but generally pandas handles structure well.
    if ext == ".csv":
        df = pd.read_csv(input_path)
    else:
        # Assuming TSV for other extensions based on your R code
        df = pd.read_csv(input_path, sep='\t')

    # 2. Exclude Identifier Column
    if id_col_name in df.columns:
        cols_to_process = [c for c in df.columns if c != id_col_name]
    else:
        warnings.warn(f"Column '{id_col_name}' not found. Processing all columns.")
        cols_to_process = df.columns.tolist()

    results_list = []

    # 3. The Loop
    for col_name in cols_to_process:
        col_data = df[col_name]
        
        # --- TYPE DETECTION LOGIC ---
        
        # A. Check if natively Numeric
        is_num_type = pd.api.types.is_numeric_dtype(col_data)
        is_date_type = False
        
        # Check if natively Date (pandas often reads as object/string initially unless parsed)
        if pd.api.types.is_datetime64_any_dtype(col_data):
            is_date_type = True
            is_num_type = False
        
        # B. If it looks like Object/String, try to convert it
        # (Pandas stores strings as 'object')
        if not is_num_type and not is_date_type and pd.api.types.is_object_dtype(col_data):
            
            # Count valid (non-null) items
            n_valid = col_data.notna().sum()
            
            if n_valid > 0:
                # 1. Try converting to Numeric
                # to_numeric with coerce turns errors into NaN
                parsed_num = pd.to_numeric(col_data, errors='coerce')
                n_parsed_num = parsed_num.notna().sum()
                
                # 2. Try converting to Date (Format YYYY-MM-DD)
                # errors='coerce' turns non-matching formats into NaT (Not a Time)
                parsed_date = pd.to_datetime(col_data, format="%Y-%m-%d", errors='coerce')
                n_parsed_date = parsed_date.notna().sum()
                
                # DECISION TREE
                if (n_parsed_num / n_valid) > 0.9:
                    col_data = parsed_num
                    is_num_type = True
                elif (n_parsed_date / n_valid) > 0.9:
                    col_data = parsed_date
                    is_date_type = True

        # --- CALCULATE STATS BASED ON TYPE ---
        
        val_str   = ""
        count_str = ""
        prop_str  = ""
        inferred_type = ""
        
        # Get valid values (drop NaN/None)
        valid_vals = col_data.dropna()

        if is_num_type:
            # --- NUMERIC LOGIC ---
            if len(valid_vals) == 0:
                val_str = "NA"; count_str = "0, 0"; prop_str = "NA"
            else:
                min_val = valid_vals.min()
                max_val = valid_vals.max()
                avg_val = valid_vals.mean()
                
                # Format: (min, mean, max)
                val_str = f"({round(min_val, 2)}, {round(avg_val, 2)}, {round(max_val, 2)})"
                
                sd_val = valid_vals.std()
                # Handle case where std is NaN (e.g. only 1 value)
                if pd.isna(sd_val): sd_val = 0.0
                
                prop_str = f"standard deviation: {round(sd_val, 2)}"
                count_str = "NA"
            
            inferred_type = "numeric"

        elif is_date_type:
            # --- DATE LOGIC ---
            if len(valid_vals) == 0:
                val_str = "NA"; count_str = "NA"; prop_str = "NA"
            else:
                min_date = valid_vals.min()
                max_date = valid_vals.max()
                
                # If these are actual timestamp objects, format them to strings
                # otherwise they might print with time (00:00:00)
                if hasattr(min_date, 'strftime'):
                    min_date = min_date.strftime('%Y-%m-%d')
                if hasattr(max_date, 'strftime'):
                    max_date = max_date.strftime('%Y-%m-%d')

                val_str = f"({min_date}, {max_date})"
                count_str = "NA"
                prop_str = "NA"
            
            inferred_type = "date"

        else:
            # --- CATEGORICAL LOGIC ---
            # value_counts returns a Series sorted by count
            # we don't want to dropna because we want to count them if they exist as a category, 
            # but usually stats exclude NaN. R's table(useNA='ifany') includes them.
            # Pandas value_counts(dropna=False) is equivalent.
            counts = col_data.value_counts(dropna=False).sort_index()
            
            # Helper to handle the "NA" label for missing values
            def format_label(x):
                return "NA" if pd.isna(x) else str(x)
            
            val_labels = [format_label(x) for x in counts.index]
            val_counts = counts.values
            
            total_n = val_counts.sum()
            proportions = val_counts / total_n
            
            val_str   = " | ".join(val_labels)
            count_str = ", ".join(map(str, val_counts))
            prop_str  = ", ".join([str(round(p, 6)) for p in proportions])
            
            inferred_type = "character"

        # 4. Construct Result Row
        results_list.append({
            "colname": col_name,
            "inferred_type": inferred_type,
            "value": val_str,
            "count": count_str,
            "proportion": prop_str
        })

    # 5. Create DataFrame from list
    stats_table = pd.DataFrame(results_list)
    return stats_table

# --- EXAMPLE USAGE ---
if __name__ == "__main__":
    # Create a dummy file for testing if needed, or point to real file
    # output = generate_column_stats("pulled_UKBAGP.tsv", "eid")
    # output.to_csv("summary_stats_python.tsv", sep='\t', index=False)
    pass