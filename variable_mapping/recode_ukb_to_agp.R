# scripts/recode_ukb_to_agp.R -----------------------------------------------
# Goal: UKB pull (pretty names) -> AGP metadata schema (AGP column names)
# - categorical fields: hard-coded mapping to 0..5
# - numeric fields: normalized to 0..5
# - array/multiselect fields: semicolon (or pipe) separated -> AGP proxies

library(readr)
library(dplyr)
library(stringr)

# ---------------------------
# 0) Helpers
# ---------------------------
na_text <- function(x) {
  x <- trimws(as.character(x))
  x[x %in% c("", "NA", "NaN", "NULL", "null",
             "Do not know", "Prefer not to answer",
             "Prefer not to answer ", "Do not know ")] <- NA
  return(x)
}

na_numeric_ukb <- function(x, neg_is_na = TRUE) {
  v <- suppressWarnings(as.numeric(trimws(as.character(x))))
  if (neg_is_na) {
    v[v < 0] <- NA
  }
  return(v)
}

check_unmapped <- function(x, recode_vec, varname) {
  x2 <- na_text(x)
  bad <- setdiff(unique(x2[!is.na(x2)]), names(recode_vec))
  if (length(bad) > 0) {
    warning(sprintf("Unmapped values in %s: %s",
                    varname, paste(bad, collapse = " | ")))
    return(FALSE)
  }
  return(TRUE)
}

# Parse amount-ish fields: "half", "6+", "4+", "3+", "1", "2", ...
parse_amount <- function(x) {
  x <- na_text(x)
  x <- tolower(x)
  x[x == "half"] <- "0.5"
  x <- gsub("\\+$", "", x)
  out <- suppressWarnings(as.numeric(x))
  return(out)
}

# Min-max scaling to 0..5 (with winsor/cap)
minmax_to_0_5 <- function(x, lower, upper) {
  v <- x
  v <- pmax(v, lower)
  v <- pmin(v, upper)

  if (upper == lower) {
    return(rep(NA_integer_, length(v)))
  }

  s <- round((v - lower) / (upper - lower) * 5)
  s[is.na(x)] <- NA
  return(as.integer(s))
}

# Quantile-binning to 0..5 (good for skewed numeric like sugar)
# keep zeros as 0; positive values split into 1..5
quantile_0_5 <- function(x) {
  v <- na_numeric_ukb(x, neg_is_na = TRUE)

  out <- rep(NA_integer_, length(v))
  out[!is.na(v) & v == 0] <- 0L

  nz <- v[!is.na(v) & v > 0]
  if (length(nz) < 50) {
    return(out)
  }

  qs <- quantile(nz, probs = seq(0, 1, 0.2), na.rm = TRUE, type = 2)
  qs <- unique(qs)
  if (length(qs) < 2) {
    return(out)
  }

  idx <- which(!is.na(v) & v > 0)
  out[idx] <- as.integer(cut(v[idx], breaks = qs, include.lowest = TRUE, labels = FALSE))
  out[idx] <- pmin(out[idx], 5L)

  return(out)
}

# Safe column getter (returns NA vector if missing)
get_col <- function(df, nm) {
  if (nm %in% names(df)) {
    return(df[[nm]])
  } else {
    return(rep(NA, nrow(df)))
  }
}

# Split multiselect arrays (UKB sometimes stored like "A; B; C" or "A | B | C")
split_multiselect <- function(s) {
  s <- trimws(as.character(s))
  if (is.na(s) || s == "" || s %in% c("NA", "NaN", "NULL", "null")) {
    return(character(0))
  }
  out <- trimws(unlist(strsplit(s, "\\s*(;|\\|)\\s*")))
  out <- out[out != ""]
  return(out)
}

date_present_0_1 <- function(x, sentinel_min = "1902-02-02") {
  # UKB "Date first reported" often uses a very early sentinel date; treat <= sentinel as "no disease"
  s <- trimws(as.character(x))
  d <- suppressWarnings(as.Date(s))

  sentinel <- as.Date(sentinel_min)
  d[!is.na(d) & d <= sentinel] <- NA

  out <- ifelse(is.na(d), 0L, 1L)
  return(as.integer(out))
}

present_nonmissing_0_1 <- function(x) {
  # for "source of report" style vars: any non-missing => 1, else 0
  xx <- na_text(x)
  out <- ifelse(is.na(xx), 0L, 1L)
  return(as.integer(out))
}


# 1) Hard-coded mappers
map_weekly_diet_0_5 <- function(x) {
  x <- na_text(x)
  recode <- c(
    "Never"                 = 0,
    "Less than once a week" = 1,
    "Once a week"           = 2,
    "2-4 times a week"      = 3,
    "5-6 times a week"      = 4,
    "Once or more daily"    = 5
  )
  check_unmapped(x, recode, "weekly_diet")
  return(as.integer(recode[x]))
}

map_alcohol_1558_0_5 <- function(x) {
  x <- na_text(x)
  recode <- c(
    "Never"                      = 0,
    "Special occasions only"     = 1,
    "One to three times a month" = 2,
    "Once or twice a week"       = 3,
    "Three or four times a week" = 4,
    "Daily or almost daily"      = 5
  )
  check_unmapped(x, recode, "alcohol_1558")
  return(as.integer(recode[x]))
}

map_smoking_20116_0_5 <- function(x) {
  x <- na_text(x)
  recode <- c(
    "Never"    = 0,
    "Previous" = 2,
    "Current"  = 5
  )
  check_unmapped(x, recode, "smoking_20116")
  return(as.integer(recode[x]))
}

map_salt_1478_0_5 <- function(x) {
  x <- na_text(x)
  recode <- c(
    "Never/rarely" = 0,
    "Sometimes"    = 2,
    "Usually"      = 4,
    "Always"       = 5
  )
  check_unmapped(x, recode, "salt_1478")
  return(as.integer(recode[x]))
}

map_bread_1448_wholegrain_0_5 <- function(x) {
  x <- na_text(x)
  recode <- c(
    "White"                   = 0,
    "Other type of bread"     = 2,
    "Brown"                   = 3,
    "Wholemeal or wholegrain" = 5
  )
  check_unmapped(x, recode, "bread_1448")
  return(as.integer(recode[x]))
}

map_milk_1418_substitute_0_5 <- function(x) {
  x <- na_text(x)
  recode <- c(
    "Soya"                   = 5,
    "Other type of milk"     = 5,
    "Never/rarely have milk" = 0,
    "Full cream"             = 0,
    "Semi-skimmed"           = 0,
    "Skimmed"                = 0
  )
  check_unmapped(x, recode, "milk_1418")
  return(as.integer(recode[x]))
}

map_sex_31_agp <- function(x) {
  x <- na_text(x)
  xs <- tolower(x)
  out <- ifelse(xs == "male", 0L, ifelse(xs == "female", 1L, NA_integer_))
  return(out)
}

map_country_of_birth_1647 <- function(x) {
  x <- na_text(x)
  out <- x
  out[x %in% c("England", "Scotland", "Wales", "Northern Ireland")] <- "United Kingdom"
  out[x == "Republic of Ireland"] <- "Ireland"
  out[x == "Elsewhere"] <- "Other"
  return(out)
}

map_race_ukb_to_agp <- function(x) {
  x <- trimws(as.character(x))

  out <- rep(NA_character_, length(x))
  out[is.na(x) | x == "" | x == "NA"] <- "Not provided"

  # CHANGE: collapse Unspecified into Other
  out[x %in% c("Do not know", "Prefer not to answer")] <- "Other"

  idx <- is.na(out)
  xs <- tolower(x[idx])
  out_idx <- which(idx)

  cauc <- xs %in% tolower(c("British", "Irish", "White", "Any other white background"))
  asian <- xs %in% tolower(c("Indian", "Pakistani", "Bangladeshi", "Chinese",
                             "Asian or Asian British", "Any other Asian background"))
  black <- xs %in% tolower(c("African", "Caribbean", "Black or Black British", "Any other Black background"))
  hisp <- grepl("hispanic|latino|latina|latin", xs)
  other <- xs %in% tolower(c("Mixed", "Any other mixed background",
                             "White and Asian", "White and Black African", "White and Black Caribbean",
                             "Other ethnic group", "Any other ethnic group"))

  out[out_idx[cauc]]  <- "Caucasian"
  out[out_idx[asian]] <- "Asian or Pacific Islander"
  out[out_idx[black]] <- "African American"
  out[out_idx[hisp]]  <- "Hispanic"
  out[out_idx[other]] <- "Other"

  out[is.na(out)] <- "Other"
  return(out)
}

binary_present <- function(x) {
  x <- na_text(x)
  out <- ifelse(is.na(x), 0L, 1L)
  return(out)
}


# 1b) NEW: multiselect -> AGP proxies

# Mouth/teeth dental problems (6149) -> teethbrushing_frequency (0..5)
# More problems => lower brushing score.
map_dental_to_teethbrushing_0_5 <- function(x) {
  x <- trimws(as.character(x))
  out <- rep(NA_integer_, length(x))

  for (i in seq_along(x)) {
    s <- x[i]

    if (is.na(s) || s == "" || s %in% c("NA","NaN","NULL","null")) {
      out[i] <- NA_integer_
      next
    }
    if (grepl("Prefer not to answer", s, fixed = TRUE) ||
        grepl("Do not know", s, fixed = TRUE)) {
      out[i] <- NA_integer_
      next
    }

    items <- split_multiselect(s)

    # if any token is prefer-not / do-not-know, treat as NA
    if (any(items %in% c("Prefer not to answer", "Do not know"))) {
      out[i] <- NA_integer_
      next
    }

    # count problems (exclude the "None" token)
    items <- items[!items %in% c("None of the above")]
    k <- length(unique(items))

    # 0 problems -> 5, 1 -> 4, ..., 5+ -> 0
    score <- 5 - k
    score <- max(0, min(5, score))
    out[i] <- as.integer(score)
  }

  return(out)
}

# flossing_frequency: slightly stricter than brushing (proxy)
map_dental_to_flossing_0_5 <- function(x) {
  brush <- map_dental_to_teethbrushing_0_5(x)
  floss <- rep(NA_integer_, length(brush))
  floss[!is.na(brush)] <- pmax(0L, brush[!is.na(brush)] - 1L)
  return(as.integer(floss))
}

# Cooking fat/oil (20090) -> olive_oil (0..5)
# Multiselect: take the MAX score (best oil used).
map_cookingfat_to_oliveoil_0_5 <- function(x) {
  x <- trimws(as.character(x))
  out <- rep(NA_integer_, length(x))

  score_map <- c(
    "Olive oil" = 5,

    "Olive spread" = 4,
    "Normal fat olive spread" = 4,
    "Very low fat olive spread" = 4,
    "Cholesterol lowering olive spread" = 4,
    "Rapeseed oil" = 4,

    "Sunflower oil" = 3,
    "Vegetable oil" = 3,
    "Other oil" = 3,
    "Polyunsaturated margarine" = 3,
    "Low fat polyunsaturated margarine" = 3,
    "Normal fat polyunsaturated margarine" = 3,
    "Very low fat polyunsaturated margarine" = 3,
    "Cholesterol lowering polyunsaturated margarine" = 3,
    "Soya margarine" = 3,
    "Low fat soya margarine" = 3,
    "Normal fat soya margarine" = 3,
    "Very low fat soya margarine" = 3,
    "Cholesterol lowering soya margarine" = 3,
    "Cholesterol lowering dairy spread" = 3,

    "Cooking fat unknown" = 2,
    "Unknown/other soft margarine" = 2,
    "Other type fat" = 2,
    "Unknown soft margarine" = 2,
    "Unknown polyunsaturated margarine" = 2,
    "Unknown olive spread" = 2,
    "Unknown soya margarine" = 2,
    "Unknown dairy spread" = 2,
    "Low fat soft margarine" = 2,
    "Normal fat soft margarine" = 2,
    "Very low fat soft margarine" = 2,
    "Cholesterol lowering soft margarine" = 2,
    "Low fat dairy spread" = 2,
    "Very low fat dairy spread" = 2,

    "Butter" = 1,
    "Normal fat butter" = 1,
    "Unknown fat butter" = 1,
    "Spreadable butter" = 1,
    "Low fat butter" = 1,
    "Lard" = 1,
    "Dairy spread" = 1,
    "Normal fat dairy spread" = 1,
    "Hard margarine" = 1
  )

  for (i in seq_along(x)) {
    s <- x[i]

    if (is.na(s) || s == "" || s %in% c("NA","NaN","NULL","null")) {
      out[i] <- NA_integer_
      next
    }
    if (grepl("Prefer not to answer", s, fixed = TRUE) ||
        grepl("Do not know", s, fixed = TRUE)) {
      out[i] <- NA_integer_
      next
    }

    items <- split_multiselect(s)

    # score each selected item (handle truncated label like "Low fat olive spread for ...")
    scores <- rep(NA_real_, length(items))
    for (j in seq_along(items)) {
      it <- items[j]
      if (it %in% names(score_map)) {
        scores[j] <- score_map[[it]]
      } else if (startsWith(it, "Low fat olive spread")) {
        scores[j] <- 4
      } else {
        scores[j] <- NA_real_
      }
    }

    if (all(is.na(scores))) {
      out[i] <- NA_integer_
      next
    }

    out[i] <- as.integer(round(max(scores, na.rm = TRUE)))
  }

  return(out)
}

# 2) Main recoder: UKB -> AGP schema
recode_ukb_to_agp <- function(ukb_df) {
  c_fluid  <- "Fluid intelligence score (FieldID: 20016)"
  c_rtmean <- "Mean time to correctly identify matches (FieldID: 20023)"

  # Source columns
  c_alcohol <- "Alcohol intake frequency. (FieldID: 1558)"
  c_smoke   <- "Smoking status (FieldID: 20116)"
  c_actdays <- "Number of days/week of moderate physical activity 10+ minutes (FieldID: 884)"
  c_fruit   <- "Fresh fruit intake (FieldID: 1309)"
  c_veg     <- "Cooked vegetable intake (FieldID: 1289)"
  c_fish    <- "Oily fish intake (FieldID: 1329)"
  c_poultry <- "Poultry intake (FieldID: 1359)"
  c_beef    <- "Beef intake (FieldID: 1369)"
  c_proc    <- "Processed meat intake (FieldID: 1349)"
  c_cheese  <- "Cheese intake (FieldID: 1408)"
  c_milk    <- "Milk type used (FieldID: 1418)"
  c_bread   <- "Bread type (FieldID: 1448)"
  c_salt    <- "Salt added to food (FieldID: 1478)"
  c_water   <- "Drinking water intake (FieldID: 100150)"
  c_lcd     <- "Low calorie drink intake (FieldID: 100160)"
  c_cob     <- "Country of birth (UK/elsewhere) (FieldID: 1647)"
  c_sex     <- "Sex (FieldID: 31)"
  c_age     <- "Age when attended assessment centre (FieldID: 21003)"
  c_bmi     <- "Body mass index (BMI) (FieldID: 21001)"
  c_weight  <- "Weight (FieldID: 21002)"
  c_append  <- "Source of report of K35 (acute appendicitis) (FieldID: 131605)"
  c_lactose <- "Date E73 first reported (lactose intolerance) (FieldID: 130804)"
  c_pizza   <- "Pizza intake (FieldID: 102000)"
  c_yogurt  <- "Yogurt intake (FieldID: 102090)"
  c_sugdrink<- "Sugar-sweetened beverages and other sugary drinks (FieldID: 26127)"
  c_freesug <- "Free sugar  (FieldID: 26012)"
  c_eggs    <- "Other egg intake (FieldID: 102980)"
  c_race    <- "Ethnic background (FieldID: 21000)"

  c_teeth   <- "Mouth/teeth dental problems (FieldID: 6149)"
  c_oil     <- "Type of fat/oil used in cooking (FieldID: 20090)"

  c_ibs_date   <- "Date K58 first reported (irritable bowel syndrome) (FieldID: 131638)"
  c_crohns_src <- "Source of report of K50 (crohn's disease [regional enteritis]) (FieldID: 131627)"
  c_uc_date    <- "Date K51 first reported (ulcerative colitis) (FieldID: 131628)"

  c_f00_date <- "Date F00 first reported (dementia in alzheimer's disease) (FieldID: 130836)"
  c_f01_date <- "Date F01 first reported (vascular dementia) (FieldID: 130838)"
  c_f02_date <- "Date F02 first reported (dementia in other diseases classified elsewhere) (FieldID: 130840)"
  c_f03_date <- "Date F03 first reported (unspecified dementia) (FieldID: 130842)"


  # Extract + convert numeric raw
  act_days <- na_numeric_ukb(get_col(ukb_df, c_actdays), neg_is_na = TRUE)
  fruit_n  <- na_numeric_ukb(get_col(ukb_df, c_fruit),   neg_is_na = TRUE)
  veg_n    <- na_numeric_ukb(get_col(ukb_df, c_veg),     neg_is_na = TRUE)
  age_n    <- na_numeric_ukb(get_col(ukb_df, c_age),     neg_is_na = TRUE)
  bmi_n    <- na_numeric_ukb(get_col(ukb_df, c_bmi),     neg_is_na = TRUE)
  wt_n     <- na_numeric_ukb(get_col(ukb_df, c_weight),  neg_is_na = TRUE)

  water_n  <- parse_amount(get_col(ukb_df, c_water))
  lcd_n    <- parse_amount(get_col(ukb_df, c_lcd))
  pizza_n  <- parse_amount(get_col(ukb_df, c_pizza))
  yogurt_n <- parse_amount(get_col(ukb_df, c_yogurt))
  eggs_n   <- parse_amount(get_col(ukb_df, c_eggs))
  
  # Normalize numeric -> 0..5
  act_0_5   <- minmax_to_0_5(act_days, lower = 0, upper = 7)
  fruit_0_5 <- minmax_to_0_5(pmin(fruit_n, 10), lower = 0, upper = 10)
  veg_0_5   <- minmax_to_0_5(pmin(veg_n, 10),   lower = 0, upper = 10)

  water_0_5 <- minmax_to_0_5(pmin(water_n, 6), lower = 0, upper = 6)
  lcd_0_5   <- minmax_to_0_5(pmin(lcd_n, 6),   lower = 0, upper = 6)

  pizza_0_5  <- minmax_to_0_5(pmin(pizza_n, 4),  lower = 0, upper = 4)
  yogurt_0_5 <- minmax_to_0_5(pmin(yogurt_n, 3), lower = 0, upper = 3)
  eggs_0_5   <- minmax_to_0_5(pmin(eggs_n, 3),   lower = 0, upper = 3)

  sugdrink_0_5 <- quantile_0_5(get_col(ukb_df, c_sugdrink))
  freesug_0_5  <- quantile_0_5(get_col(ukb_df, c_freesug))

  # Categorical -> 0..5
  alcohol_0_5 <- map_alcohol_1558_0_5(get_col(ukb_df, c_alcohol))
  smoke_0_5   <- map_smoking_20116_0_5(get_col(ukb_df, c_smoke))

  fish_0_5   <- map_weekly_diet_0_5(get_col(ukb_df, c_fish))
  poultry_0_5 <- map_weekly_diet_0_5(get_col(ukb_df, c_poultry))
  beef_0_5   <- map_weekly_diet_0_5(get_col(ukb_df, c_beef))
  proc_0_5   <- map_weekly_diet_0_5(get_col(ukb_df, c_proc))
  cheese_0_5 <- map_weekly_diet_0_5(get_col(ukb_df, c_cheese))

  milk_sub_0_5   <- map_milk_1418_substitute_0_5(get_col(ukb_df, c_milk))
  wholegrain_0_5 <- map_bread_1448_wholegrain_0_5(get_col(ukb_df, c_bread))
  salt_0_5       <- map_salt_1478_0_5(get_col(ukb_df, c_salt))

  # Binary proxies
  lactose_bin  <- binary_present(get_col(ukb_df, c_lactose))
  appendix_bin <- binary_present(get_col(ukb_df, c_append))

  # date for cognitive/ICD
  ibs_bin    <- date_present_0_1(get_col(ukb_df, c_ibs_date))
  uc_bin     <- date_present_0_1(get_col(ukb_df, c_uc_date))
  crohns_bin <- present_nonmissing_0_1(get_col(ukb_df, c_crohns_src))

  f00_bin <- date_present_0_1(get_col(ukb_df, c_f00_date))
  f01_bin <- date_present_0_1(get_col(ukb_df, c_f01_date))
  f02_bin <- date_present_0_1(get_col(ukb_df, c_f02_date))
  f03_bin <- date_present_0_1(get_col(ukb_df, c_f03_date))


  # Alcohol consumption (AGP has 0..2)
  alcohol_consumption_0_2 <- ifelse(is.na(alcohol_0_5), NA_integer_,
                                    ifelse(alcohol_0_5 == 0, 0L,
                                           ifelse(alcohol_0_5 <= 2, 1L, 2L)))

  # NEW: multiselect-derived proxies
  teeth_0_5 <- map_dental_to_teethbrushing_0_5(get_col(ukb_df, c_teeth))
  floss_0_5 <- map_dental_to_flossing_0_5(get_col(ukb_df, c_teeth))
  olive_0_5 <- map_cookingfat_to_oliveoil_0_5(get_col(ukb_df, c_oil))
  
  # fluid score
  fluid_score <- na_numeric_ukb(get_col(ukb_df, c_fluid),  neg_is_na = TRUE)
  mean_rt_ms  <- na_numeric_ukb(get_col(ukb_df, c_rtmean), neg_is_na = TRUE)

  out <- tibble(
    sample_name = as.character(get_col(ukb_df, "eid")),
    country = "United Kingdom",
    race = map_race_ukb_to_agp(get_col(ukb_df, c_race)),
    sex = map_sex_31_agp(get_col(ukb_df, c_sex)),
    country_of_birth = map_country_of_birth_1647(get_col(ukb_df, c_cob)),

    age_corrected = age_n,
    bmi = bmi_n,
    weight_kg = wt_n,

    alcohol_frequency = alcohol_0_5,
    alcohol_consumption = alcohol_consumption_0_2,

    smoking_frequency = smoke_0_5,
    exercise_frequency = act_0_5,

    fruit_frequency = fruit_0_5,
    vegetable_frequency = veg_0_5,

    seafood_frequency = fish_0_5,
    red_meat_frequency = beef_0_5,
    high_fat_red_meat_frequency = proc_0_5,
    milk_cheese_frequency = cheese_0_5,
    poultry_frequency = poultry_0_5,

    milk_substitute_frequency = milk_sub_0_5,
    whole_grain_frequency = wholegrain_0_5,
    salted_snacks_frequency = salt_0_5,

    one_liter_of_water_a_day_frequency = water_0_5,
    artificial_sweeteners = lcd_0_5,

    prepared_meals_frequency = pizza_0_5,
    ready_to_eat_meals_frequency = pizza_0_5,
    probiotic_frequency = yogurt_0_5,
    whole_eggs = eggs_0_5,

    sugar_sweetened_drink_frequency = sugdrink_0_5,
    free_sugar_scaled_0_5 = freesug_0_5,

    lactose = lactose_bin,
    appendix_removed = appendix_bin,

    # NEW AGP-style additions
    teethbrushing_frequency = teeth_0_5,
    flossing_frequency = floss_0_5,
    olive_oil = olive_0_5,
    fluid_intelligence_score = fluid_score,
    mean_match_rt_ms = mean_rt_ms,
    
    # for ICD 10
    ibs = ibs_bin,
    crohns_disease = crohns_bin,
    ulcerative_colitis = uc_bin,

    dementia_alzheimers = f00_bin,
    dementia_vascular   = f01_bin,
    dementia_other      = f02_bin,
    dementia_unspecified= f03_bin,
  )

  return(out)
}

# 3) Run: read UKB pull and write AGP-format file
ukb <- read_tsv("pulled_UKBAGP.tsv", show_col_types = FALSE)
ukb_agp <- recode_ukb_to_agp(ukb)
write_csv(ukb_agp, "ukb_as_agp_metadata.csv")
