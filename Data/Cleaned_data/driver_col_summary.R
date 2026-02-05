source("ultimate_summary.R")

cols_to_ignore <- c("survey_id", "bmi_corrected", "state", "longitude", "latitude", "elevation", "country_of_birth")

# --- EXAMPLE USAGE ---
output <- generate_column_stats("processed_metadata.csv", "sample_name")
write_tsv(output, "summary_stats_metadata.tsv")

output <- generate_column_stats("AGP_Metadata.csv", "sample_name", cols_to_skip=cols_to_ignore)
write_tsv(output, "more_columns_summary.tsv")