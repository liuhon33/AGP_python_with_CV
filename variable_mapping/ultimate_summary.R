library(tidyverse)
library(tools)

generate_column_stats <- function(input_path, id_col_name) {
  
  # 1. Read Data
  ext <- file_ext(input_path)
  if (tolower(ext) == "csv") {
    df <- read_csv(input_path, show_col_types = FALSE, name_repair = "unique")
  } else {
    df <- read_tsv(input_path, show_col_types = FALSE, name_repair = "unique")
  }
  
  # 2. Exclude Identifier Column
  if (id_col_name %in% names(df)) {
    cols_to_process <- setdiff(names(df), id_col_name)
  } else {
    warning(paste("Column '", id_col_name, "' not found. Processing all columns."))
    cols_to_process <- names(df)
  }
  
  # 3. Initialize list for results
  results_list <- list()
  
  # 4. The Loop
  for (col_name in cols_to_process) {
    
    col_data <- df[[col_name]]
    
    # --- TYPE DETECTION LOGIC ---
    
    # A. Check if natively Numeric or Date
    is_num_type <- is.numeric(col_data) || is.integer(col_data)
    is_date_type <- inherits(col_data, "Date")
    is_array_type <- FALSE # Reset
    
    # If Date was detected natively, suppress num type
    if (is_date_type) is_num_type <- FALSE
    
    # B. If it looks like Character, try to convert it OR detect Array
    if (!is_num_type && !is_date_type && is.character(col_data)) {
      
      n_valid <- sum(!is.na(col_data))
      
      # 1. Try converting to Numeric
      parsed_num <- suppressWarnings(as.numeric(col_data))
      n_parsed_num <- sum(!is.na(parsed_num))
      
      # 2. Try converting to Date
      parsed_date <- as.Date(col_data, format = "%Y-%m-%d")
      n_parsed_date <- sum(!is.na(parsed_date))
      
      # 3. Check for Semicolons (Array detection)
      # We check valid strings to see if they contain ";"
      has_semicolon <- FALSE
      if (n_valid > 0) {
        # Check a sample or all non-na values for semicolon
        valid_vals <- col_data[!is.na(col_data)]
        if (any(grepl(";", valid_vals))) {
          has_semicolon <- TRUE
        }
      }
      
      # --- DECISION TREE ---
      if (n_valid > 0 && (n_parsed_num / n_valid) > 0.9) {
        # It's a Number disguised as text
        col_data <- parsed_num
        is_num_type <- TRUE
      } else if (n_valid > 0 && (n_parsed_date / n_valid) > 0.9) {
        # It's a Date disguised as text
        col_data <- parsed_date
        is_date_type <- TRUE
      } else if (has_semicolon) {
        # It is a multiple-choice array
        is_array_type <- TRUE
      }
    }
    
    # --- CALCULATE STATS BASED ON TYPE ---
    
    val_str   <- ""
    count_str <- ""
    prop_str  <- ""
    inferred_type <- ""
    
    if (is_num_type) {
      # --- NUMERIC LOGIC ---
      valid_vals <- col_data[!is.na(col_data)]
      if (length(valid_vals) == 0) {
        val_str <- "NA"; count_str <- "0, 0"; prop_str <- "NA"
      } else {
        min_val <- min(valid_vals)
        mean_val <- mean(valid_vals)
        max_val <- max(valid_vals)
        val_str <- paste0("(", round(min_val, 2), ", ", round(mean_val,2), ", ", round(max_val, 2), ")")
        
        sd_val <- sd(valid_vals)
        prop_str <- paste0("sd: ", round(sd_val, 2))
        count_str <- "NA"
      }
      inferred_type <- "numeric"
      
    } else if (is_date_type) {
      # --- DATE LOGIC ---
      valid_vals <- col_data[!is.na(col_data)]
      if (length(valid_vals) == 0) {
        val_str <- "NA"; count_str <- "NA"; prop_str <- "NA"
      } else {
        val_str <- paste0("(", min(valid_vals), ", ", max(valid_vals), ")")
        count_str <- "NA"; prop_str <- "NA"
      }
      inferred_type <- "date"
      
    } else if (is_array_type) {
      # --- ARRAY LOGIC (New) ---
      
      # Get valid rows (to calculate proportion of RESPONDENTS, not just tags)
      valid_vals <- col_data[!is.na(col_data)]
      n_respondents <- length(valid_vals) 
      
      # 1. Split strings by semicolon
      # 2. Unlist to get one big vector of tags
      # 3. Trim whitespace (e.g. " A" becomes "A")
      all_tokens <- unlist(strsplit(as.character(valid_vals), ";"))
      all_tokens <- trimws(all_tokens)
      all_tokens <- all_tokens[all_tokens != ""] # Remove empty tokens
      
      if (length(all_tokens) == 0) {
        val_str <- "Empty"; count_str <- "0"; prop_str <- "0"
      } else {
        # Count individual tags
        counts_df <- as.data.frame(table(all_tokens))
        colnames(counts_df) <- c("val", "n")
        
        # Calculate Prop: (Count of Tag) / (Total Number of Non-NA Rows)
        # This gives the "prevalence" of the choice.
        # Note: These will sum to > 100% because choices are not mutually exclusive.
        counts_df$prop <- counts_df$n / n_respondents
        
        # Sort by frequency (highest first)
        counts_df <- counts_df[order(-counts_df$n), ]
        
        val_str   <- paste(counts_df$val, collapse = " | ")
        count_str <- paste(counts_df$n, collapse = ", ")
        prop_str  <- paste(round(counts_df$prop, 6), collapse = ", ")
      }
      inferred_type <- "array"
      
    } else {
      # --- STANDARD CATEGORICAL LOGIC ---
      counts_df <- as.data.frame(table(col_data, useNA = "ifany"))
      colnames(counts_df) <- c("val", "n")
      counts_df$prop <- counts_df$n / sum(counts_df$n)
      
      # Sort by frequency
      counts_df <- counts_df[order(-counts_df$n), ]
      
      val_label <- as.character(counts_df$val)
      val_label[is.na(val_label)] <- "NA"
      
      val_str   <- paste(val_label, collapse = " | ")
      count_str <- paste(counts_df$n, collapse = ", ")
      prop_str  <- paste(round(counts_df$prop, 6), collapse = ", ")
      
      inferred_type <- "character"
    }
    
    # 5. Store Result
    results_list[[col_name]] <- tibble(
      colname       = col_name,
      inferred_type = inferred_type,
      value         = val_str,
      count         = count_str,
      proportion    = prop_str
    )
  }
  
  # 6. Bind all results
  stats_table <- bind_rows(results_list)
  return(stats_table)
}

# --- EXAMPLE USAGE ---
output <- generate_column_stats("ukb_as_agp_metadata.csv", "sample_name")
write_tsv(output, "summary_stats.tsv")