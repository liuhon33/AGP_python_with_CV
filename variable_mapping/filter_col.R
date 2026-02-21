# ----------------------------
# Packages (visual + fast ops)
# ----------------------------
pkgs <- c("data.table", "ggplot2", "naniar", "dplyr")
to_install <- pkgs[!pkgs %in% rownames(installed.packages())]
if (length(to_install)) install.packages(to_install)

library(data.table)
library(ggplot2)
library(naniar)
library(dplyr)

# ---------------------------------------------
# Greedy recursive complete-case filtering
# ---------------------------------------------
greedy_complete_filter <- function(df,
                                  cols = names(df),
                                  id_cols = character(0),
                                  stop_when_no_na_left = TRUE) {
  dt <- as.data.table(df)

  # columns we will consider in the greedy ordering
  cols <- setdiff(cols, id_cols)
  cols <- cols[cols %in% names(dt)]

  keep <- rep(TRUE, nrow(dt))
  steps <- list()
  step_i <- 0L

  while (length(cols) > 0 && any(keep)) {
    step_i <- step_i + 1L

    # NA counts within the currently retained rows
    na_counts <- vapply(cols, function(cc) sum(is.na(dt[[cc]][keep])), integer(1))

    pick <- names(which.min(na_counts))[1]
    n_before <- sum(keep)
    na_in_pick <- na_counts[[pick]]

    keep2 <- keep & !is.na(dt[[pick]])
    n_after <- sum(keep2)

    steps[[step_i]] <- data.table(
      step = step_i,
      field = pick,
      n_before = n_before,
      n_after = n_after,
      dropped = n_before - n_after,
      na_in_field_before = na_in_pick,
      prop_remaining = n_after / nrow(dt)
    )

    keep <- keep2
    cols <- setdiff(cols, pick)

    # optional early stop: remaining rows have no NAs in remaining columns
    if (stop_when_no_na_left && length(cols) > 0 && any(keep)) {
      na_left <- vapply(cols, function(cc) sum(is.na(dt[[cc]][keep])), integer(1))
      if (all(na_left == 0L)) {
        # append the remaining columns (no further attrition) just as ordering info
        extra <- data.table(
          step = (step_i + 1L):(step_i + length(cols)),
          field = cols,
          n_before = sum(keep),
          n_after  = sum(keep),
          dropped = 0L,
          na_in_field_before = 0L,
          prop_remaining = sum(keep) / nrow(dt)
        )
        steps[[length(steps) + 1L]] <- extra
        break
      }
    }
  }

  steps_dt <- rbindlist(steps, fill = TRUE)
  list(
    steps = steps_dt,
    row_mask = keep,
    filtered = dt[keep],
    chosen_order = steps_dt$field
  )
}

# ---------------------------------------------
# Plot helpers (attrition + missingness overview)
# ---------------------------------------------
plot_attrition <- function(steps_dt) {
  steps_dt2 <- steps_dt %>%
    mutate(field = factor(field, levels = field)) %>%
    arrange(step)

  p1 <- ggplot(steps_dt2, aes(x = step, y = n_after)) +
    geom_line() +
    geom_point() +
    scale_x_continuous(breaks = steps_dt2$step) +
    labs(x = "Greedy step", y = "Rows remaining", title = "Attrition as you add non-missing constraints")

  p2 <- ggplot(steps_dt2, aes(x = reorder(field, step), y = dropped)) +
    geom_col() +
    coord_flip() +
    labs(x = "Field chosen (in order)", y = "Rows dropped at this step", title = "Drop in N at each greedy step")

  list(remaining_curve = p1, drop_bars = p2)
}

plot_missing_overview <- function(df, sample_n = 5000) {
  # UKB is huge; sample for heatmap-style viz
  dsmall <- if (nrow(df) > sample_n) dplyr::slice_sample(df, n = sample_n) else df

  p_miss <- naniar::vis_miss(dsmall, cluster = TRUE) +
    ggtitle(paste0("Missingness heatmap (sample n=", nrow(dsmall), ")"))

  p_upset <- naniar::gg_miss_upset(dsmall) +
    ggtitle("Missingness pattern UpSet plot (sampled)")

  list(heatmap = p_miss, upset = p_upset)
}

----------------------------
Example usage
----------------------------
df <- your UKB data.frame in memory
res <- greedy_complete_filter(df, id_cols = c("eid"))  # add your ID cols if any

View the greedy ordering / attrition table
res$steps

Get the fully filtered dataset after the greedy process
df_greedy_complete <- res$filtered

Plots
attr_plots <- plot_attrition(res$steps)
print(attr_plots$remaining_curve)
print(attr_plots$drop_bars)

Missingness visualizations (sampling for speed)
miss_plots <- plot_missing_overview(df)
print(miss_plots$heatmap)
print(miss_plots$upset)