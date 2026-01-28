# utils/summarise_fields.R ---------------------------------------------------
summarise_fields <- function(df,
                             id_cols = c("eid", "instance"),
                             max_levels_to_print = 50,
                             top_n_levels = 30) {
  stopifnot(is.data.frame(df))

  n_total <- nrow(df)
  cols <- setdiff(names(df), id_cols)

  parse_field <- function(colname) {
    # Example: "Alcohol intake frequency. (FieldID: 1558)"
    field_id <- sub(".*\\(FieldID:\\s*([0-9]+)\\).*", "\\1", colname)
    if (identical(field_id, colname)) field_id <- NA_character_

    label <- sub("\\s*\\(FieldID:.*\\)\\s*$", "", colname)
    list(field_id = field_id, label = label)
  }

  is_date_like <- function(x) {
    if (inherits(x, "Date")) return(TRUE)
    if (is.character(x)) {
      # yyyy-mm-dd
      return(any(grepl("^\\d{4}-\\d{2}-\\d{2}$", x[!is.na(x)][1:min(50, sum(!is.na(x)))])))
    }
    FALSE
  }

  # Helper: summarize one column
  summarize_one <- function(x, colname) {
    meta <- parse_field(colname)

    # Normalize types a bit (but don't destroy the original)
    cls <- class(x)[1]
    n_missing <- sum(is.na(x))
    n_nonmiss <- n_total - n_missing

    # Identify "categorical-ish" vs numeric
    # Note: integers with small #unique are often categorical codes.
    x_nonmiss <- x[!is.na(x)]
    n_unique <- if (n_nonmiss == 0) 0 else length(unique(x_nonmiss))

    date_flag <- is_date_like(x)
    numeric_flag <- is.numeric(x) && !date_flag

    # numeric summary
    num_min <- num_q1 <- num_med <- num_mean <- num_q3 <- num_max <- NA_real_
    if (numeric_flag && n_nonmiss > 0) {
      qs <- stats::quantile(x_nonmiss, probs = c(0, 0.25, 0.5, 0.75, 1), na.rm = TRUE, type = 2)
      num_min  <- as.numeric(qs[1])
      num_q1   <- as.numeric(qs[2])
      num_med  <- as.numeric(qs[3])
      num_q3   <- as.numeric(qs[4])
      num_max  <- as.numeric(qs[5])
      num_mean <- mean(x_nonmiss, na.rm = TRUE)
    }

    # date range (if date-like)
    date_min <- date_max <- NA_character_
    if (date_flag && n_nonmiss > 0) {
      if (!inherits(x_nonmiss, "Date")) {
        # try parse yyyy-mm-dd
        x_parsed <- suppressWarnings(as.Date(as.character(x_nonmiss)))
      } else {
        x_parsed <- x_nonmiss
      }
      x_parsed <- x_parsed[!is.na(x_parsed)]
      if (length(x_parsed) > 0) {
        date_min <- as.character(min(x_parsed))
        date_max <- as.character(max(x_parsed))
      }
    }

    # Decide if we should produce levels table:
    # - character/factor/logical always
    # - numeric/integer if unique count is "small" (codes)
    categorical_flag <- is.character(x) || is.factor(x) || is.logical(x) ||
      (!date_flag && numeric_flag && n_unique > 0 && n_unique <= max_levels_to_print)

    list(
      field_id = meta$field_id,
      field_label = meta$label,
      colname = colname,
      class = cls,
      n = n_total,
      n_missing = n_missing,
      pct_missing = if (n_total == 0) NA_real_ else 100 * n_missing / n_total,
      n_unique_nonmissing = n_unique,
      is_date_like = date_flag,
      is_numeric = numeric_flag,
      is_categorical_like = categorical_flag,
      num_min = num_min,
      num_q1 = num_q1,
      num_median = num_med,
      num_mean = num_mean,
      num_q3 = num_q3,
      num_max = num_max,
      date_min = date_min,
      date_max = date_max
    )
  }

  # Build field summary
  summary_list <- lapply(cols, function(nm) summarize_one(df[[nm]], nm))
  field_summary <- do.call(rbind, lapply(summary_list, as.data.frame, stringsAsFactors = FALSE))
  rownames(field_summary) <- NULL

  # Build levels table for categorical-like columns
  make_levels <- function(x, colname) {
    x2 <- x
    if (inherits(x2, "Date")) x2 <- as.character(x2)
    if (is.factor(x2)) x2 <- as.character(x2)

    x_nonmiss <- x2[!is.na(x2)]
    if (length(x_nonmiss) == 0) return(NULL)

    tab <- sort(table(x_nonmiss), decreasing = TRUE)
    # keep only top_n_levels to avoid exploding output
    tab <- head(tab, top_n_levels)

    meta <- parse_field(colname)
    data.frame(
      field_id = meta$field_id,
      field_label = meta$label,
      colname = colname,
      value = names(tab),
      count = as.integer(tab),
      proportion = as.numeric(tab) / n_total,
      stringsAsFactors = FALSE
    )
  }

  lev_rows <- lapply(cols, function(nm) {
    flag <- field_summary$is_categorical_like[field_summary$colname == nm]
    if (isTRUE(flag)) make_levels(df[[nm]], nm) else NULL
  })
  field_levels <- do.call(rbind, lev_rows)
  if (is.null(field_levels)) {
    field_levels <- data.frame(
      field_id=character(), field_label=character(), colname=character(),
      value=character(), count=integer(), proportion=numeric(),
      stringsAsFactors = FALSE
    )
  }

  list(field_summary = field_summary, field_levels = field_levels)
}
