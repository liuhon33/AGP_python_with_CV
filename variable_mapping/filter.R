# scripts/filter_missing_rows.R ---------------------------------------------
# Goal: read AGP-format UKB metadata and drop rows where >20% of columns are NA

library(readr)
library(dplyr)

infile  <- "ukb_as_agp_metadata.csv"
outfile <- "ukb_as_agp_metadata.filtered_93pct_complete.csv"

# 1) Read
df <- read_csv(infile, show_col_types = FALSE)

# 2) Choose which columns to count NA over
#    Usually you DON'T want to penalize the ID column.
id_cols <- c("sample_name")
cols_check <- setdiff(names(df), id_cols)

# 3) Compute per-row missingness (% NA across cols_check)
#    rowMeans(is.na(.)) returns fraction in [0,1]
missing_frac <- rowMeans(is.na(df[, cols_check, drop = FALSE]))

# 4) Filter: keep rows with <= 7% missing
df_filt <- df[missing_frac <= 0.07, , drop = FALSE]

# 5) Quick report
cat("Input rows:   ", nrow(df), "\n")
cat("Output rows:  ", nrow(df_filt), "\n")
cat("Dropped rows: ", nrow(df) - nrow(df_filt), "\n")
cat("Input cols:   ", ncol(df), "\n\n")

cat("Missingness fraction summary (all rows):\n")
print(summary(missing_frac))

cat("\nFirst 10 rows of filtered df:\n")
print(head(df_filt, 10))

# 6) Save filtered dataset
write_csv(df_filt, outfile)
cat("\nWrote filtered file to: ", outfile, "\n")
