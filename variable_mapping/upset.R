# scripts/upset_group_completeness.R ----------------------------------------
# Goal: build group-level completeness flags (YES/NO) and draw an UpSet plot
# Input:  variable_mapping/ukb_as_agp_metadata.csv
# Output: variable_mapping/upset_group_completeness.pdf + a CSV of flags

library(readr)
library(dplyr)
library(ggplot2)

infile  <- "ukb_as_agp_metadata.csv"
out_pdf <- "upset_group_completeness.pdf"
out_csv <- "group_completeness_flags.csv"

df <- read_csv(infile, show_col_types = FALSE)

# 1) Define groups
groups <- list(
  demo = c("sex", "age_corrected", "bmi", "weight_kg"),

  diet = c(
    "fruit_frequency", "vegetable_frequency", "seafood_frequency",
    "red_meat_frequency", "high_fat_red_meat_frequency"
  ),
  diet2 = c(
    "artificial_sweeteners", "prepared_meals_frequency",
    "ready_to_eat_meals_frequency", "probiotic_frequency",
    "whole_eggs"
  ),
  diet3 = c(
    "milk_cheese_frequency", "whole_grain_frequency",
    "salted_snacks_frequency", "one_liter_of_water_a_day_frequency"
  ),

  alcohol_smoking = c("alcohol_frequency", "alcohol_consumption", "smoking_frequency"),

  cognition = c("fluid_intelligence_score"), #, "mean_match_rt_ms"),

  oral_proxy = c("teethbrushing_frequency", "flossing_frequency", "olive_oil"),

  gi = c("ibs", "crohns_disease", "ulcerative_colitis")#,
  # dementia = c("dementia_alzheimers", "dementia_vascular", "dementia_other", "dementia_unspecified")
)

# 2) Decide what "YES" means for a group
# STRICT: require all columns present
mode <- "strict"   # "strict" or "soft"

# SOFT: require >= this fraction of columns present
min_prop_present <- 0.9

make_group_flag <- function(df, cols, mode = "strict", min_prop_present = 0.80) {
  cols2 <- intersect(cols, names(df))

  if (length(cols2) == 0) {
    # If none of the columns exist, return NA flags
    return(rep(NA, nrow(df)))
  }

  m <- as.data.frame(lapply(df[, cols2, drop = FALSE], is.na))

  if (mode == "strict") {
    # TRUE if ALL are non-NA
    flag <- !apply(m, 1, any)
  } else {
    # TRUE if >= min_prop_present are non-NA
    prop_present <- 1 - rowMeans(m)
    flag <- prop_present >= min_prop_present
  }

  return(flag)
}

# build group flag dataframe
flag_df <- lapply(names(groups), function(g) {
  make_group_flag(df, groups[[g]], mode = mode, min_prop_present = min_prop_present)
})
names(flag_df) <- names(groups)
flag_df <- as.data.frame(flag_df)

# optional: keep ID
if ("sample_name" %in% names(df)) {
  flag_df <- cbind(sample_name = df$sample_name, flag_df)
}

write_csv(flag_df, out_csv)


# 3) Make the UpSet plot
# Use ComplexUpset if available (prettier + ggplot-based); else fallback to UpSetR.
group_cols <- names(groups)

# drop groups that are completely NA (e.g., if columns missing)
valid_groups <- group_cols[sapply(flag_df[, group_cols, drop = FALSE], function(v) !all(is.na(v)))]
if (length(valid_groups) < 2) stop("Need at least 2 valid groups to make an UpSet plot.")

# Prepare plotting frame (ComplexUpset wants logicals)
plot_df <- flag_df
for (g in valid_groups) plot_df[[g]] <- as.logical(plot_df[[g]])

if (requireNamespace("ComplexUpset", quietly = TRUE)) {
  library(ComplexUpset)

  p <- ComplexUpset::upset(
    plot_df,
    intersect = valid_groups,
    base_annotations = list(
      "Intersection size" = ComplexUpset::intersection_size()
    ),
    set_sizes = ComplexUpset::upset_set_size()
  ) +
    labs(
      title = "UpSet: Group-level completeness patterns",
      subtitle = if (mode == "strict") {
        "YES = all columns in the group are non-missing"
      } else {
        paste0("YES = at least ", min_prop_present * 100, "% of columns in the group are non-missing")
      }
    )

  ggsave(out_pdf, p, width = 12, height = 7)

} else if (requireNamespace("UpSetR", quietly = TRUE)) {
  # UpSetR works with a data frame of 0/1 or TRUE/FALSE columns
  # Save to PDF via base graphics
  library(UpSetR)

  # UpSetR doesn't like NA: convert NA -> FALSE (treat as not complete)
  plot_mat <- plot_df[, valid_groups, drop = FALSE]
  for (g in valid_groups) plot_mat[[g]][is.na(plot_mat[[g]])] <- FALSE

  pdf(out_pdf, width = 12, height = 7)
  UpSetR::upset(
    plot_mat,
    sets = valid_groups,
    order.by = "freq",
    keep.order = TRUE
  )
  dev.off()

} else {
  stop("Please install either ComplexUpset or UpSetR.\nTry: install.packages('ComplexUpset') or install.packages('UpSetR')")
}

cat("Wrote:\n")
cat(" - Flags CSV: ", out_csv, "\n", sep = "")
cat(" - UpSet PDF: ", out_pdf, "\n", sep = "")

# ---------------------------
# 4) Group-level missingness/completion summary + plot
#    (add this block AFTER you build `flag_df` and `groups`)
# ---------------------------

# Choose how to score "completion"
# strict: per-row group completion = 1 only if ALL cols in group present
# soft:   per-row group completion = fraction present (0..1), then report mean
completion_mode <- "strict"   # "strict" or "soft"
min_prop_present <- 0.80      # used only if you later want a binary soft flag

# Per-row completion score for a group
group_completion_score <- function(df, cols) {
  cols2 <- intersect(cols, names(df))
  if (length(cols2) == 0) return(rep(NA_real_, nrow(df)))

  na_mat <- sapply(df[, cols2, drop = FALSE], is.na)  # n x p logical
  if (!is.matrix(na_mat)) na_mat <- matrix(na_mat, ncol = 1)

  prop_present <- 1 - rowMeans(na_mat)  # fraction non-missing in that group per row
  return(as.numeric(prop_present))
}

# Build summary table: for each group
group_summary <- lapply(names(groups), function(g) {
  cols2 <- intersect(groups[[g]], names(df))
  score <- group_completion_score(df, groups[[g]])

  # strict completion per row (all present)
  strict_flag <- ifelse(is.na(score), NA, score == 1)

  # soft completion per row (>= min_prop_present present)
  soft_flag <- ifelse(is.na(score), NA, score >= min_prop_present)

  data.frame(
    group = g,
    n_cols_defined = length(groups[[g]]),
    n_cols_found   = length(cols2),

    # average fraction present across rows
    mean_prop_present = mean(score, na.rm = TRUE),
    median_prop_present = median(score, na.rm = TRUE),

    # percent of rows fully complete (strict)
    pct_rows_strict_complete = 100 * mean(strict_flag, na.rm = TRUE),

    # percent of rows >= threshold (soft-as-binary)
    pct_rows_soft_complete = 100 * mean(soft_flag, na.rm = TRUE),

    stringsAsFactors = FALSE
  )
}) %>% bind_rows()

# Pick what you want on the y-axis
# - strict: % rows fully complete
# - soft:   mean % columns present (mean_prop_present * 100)
plot_metric <- if (completion_mode == "strict") {
  "pct_rows_strict_complete"
} else {
  "mean_prop_present"
}

plot_df <- group_summary %>%
  mutate(
    y = if (completion_mode == "strict") pct_rows_strict_complete else mean_prop_present * 100
  ) %>%
  arrange(y)

# Save summary table
out_group_missing_csv <- "group_completion_summary.csv"
readr::write_csv(group_summary, out_group_missing_csv)

# Plot
out_group_missing_pdf <- "group_completion_barplot.pdf"

p_missing <- ggplot(plot_df, aes(x = reorder(group, y), y = y)) +
  geom_col() +
  coord_flip() +
  labs(
    title = "Group-level completion",
    subtitle = if (completion_mode == "strict") {
      "Y = % of rows with ALL columns present in group"
    } else {
      "Y = average % of columns present within group (per-row, then averaged)"
    },
    x = "Group",
    y = "Percent completion"
  )

ggsave(out_group_missing_pdf, p_missing, width = 10, height = 6)

cat("Wrote:\n")
cat(" - Group completion summary CSV: ", out_group_missing_csv, "\n", sep = "")
cat(" - Group completion barplot PDF: ", out_group_missing_pdf, "\n", sep = "")
