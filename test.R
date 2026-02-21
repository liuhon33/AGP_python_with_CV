
# simple_age_cognition_regression.R
# Goal: simplest regression of cognition ~ age_corrected

library(readr)
library(dplyr)

# ---- CONFIG ----
infile <- "variable_mapping/ukb_as_agp_metadata.csv"
id_col <- "sample_name"

cognition_col <- "fluid_intelligence_score"
age_col <- "age_corrected"

# ---- LOAD ----
df <- read_csv(infile, show_col_types = FALSE)

# ---- COERCE NUMERIC + KEEP COMPLETE CASES ----
dat <- df %>%
  mutate(
    y = as.numeric(.data[[cognition_col]]),
    age = as.numeric(.data[[age_col]])
  ) %>%
  select(all_of(c(id_col, "y", "age"))) %>%
  filter(!is.na(y), !is.na(age))

cat("N used:", nrow(dat), "\n")
cat("Age range:", min(dat$age), "to", max(dat$age), "\n")
cat("Cognition range:", min(dat$y), "to", max(dat$y), "\n\n")

# ---- SIMPLE OLS: cognition ~ age ----
fit <- lm(y ~ age + age**2, data = dat)
print(summary(fit))

# Optional: robust SE (HC3) if you want to match your Python setup
if (requireNamespace("sandwich", quietly = TRUE) && requireNamespace("lmtest", quietly = TRUE)) {
  library(sandwich)
  library(lmtest)
  cat("\n--- Robust SE (HC3) ---\n")
  print(coeftest(fit, vcov. = vcovHC(fit, type = "HC3")))
} else {
  cat("\n(Optional) Install robust SE packages:\n")
  cat("install.packages(c('sandwich','lmtest'))\n")
}

