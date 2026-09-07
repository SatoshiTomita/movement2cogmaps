options(warn = -1)

library(DescTools)
library(reticulate)

np <- import("numpy", convert = FALSE)


args <- commandArgs(trailingOnly = TRUE)
BASE_DIR <- if (length(args) >= 1) args[1] else "/home/tomita/movement2cogmaps/R_stats"
OUTPUT_FILE <- if (length(args) >= 2) args[2] else file.path(BASE_DIR, "stats_out.csv")
set.seed(123)

# Extract all three BH-adjusted comparisons by group label.
get_wilcox_pvals <- function(test) {
  mat <- test$p.value
  c(mat["2", "1"], mat["3", "2"], mat["3", "1"])
}

# Initialize results data.frame
results <- data.frame(
  metric = character(),
  source = character(),
  jonckheere_p = numeric(),
  wilcox_1_2_p = numeric(),
  wilcox_2_3_p = numeric(),
  wilcox_1_3_p = numeric(),
  stringsAsFactors = FALSE
)

# ---- 1. Spatial Information for Rate Maps ----

g_model <- np$load(file.path(BASE_DIR, 'data/g_model.npy')); g_model <- py_to_r(g_model)
g_real <- np$load(file.path(BASE_DIR, 'data/g_real.npy')); g_real <- py_to_r(g_real)

# MODEL DATA
x_model <- np$load(file.path(BASE_DIR, 'data/sir_model.npy')); x_model <- py_to_r(x_model)
jt_model <- JonckheereTerpstraTest(x_model, g_model, alternative = 'increasing', nperm = 10000)
wilcox_model <- pairwise.wilcox.test(x_model, g_model, p.adjust.method = "BH", alternative = "greater")
pvals_model <- get_wilcox_pvals(wilcox_model)
results <- rbind(results, data.frame(
  metric = "SIr",
  source = "model",
  jonckheere_p = jt_model$p.value,
  wilcox_1_2_p = pvals_model[1],
  wilcox_2_3_p = pvals_model[2],
  wilcox_1_3_p = pvals_model[3]
))

# EXPERIMENTAL DATA
x_real <- np$load(file.path(BASE_DIR, 'data/sir_real.npy')); x_real <- py_to_r(x_real)
jt_real <- JonckheereTerpstraTest(x_real, g_real, alternative = 'increasing', nperm = 10000)
wilcox_real <- pairwise.wilcox.test(x_real, g_real, p.adjust.method = "BH", alternative = "greater")
pvals_real <- get_wilcox_pvals(wilcox_real)
results <- rbind(results, data.frame(
  metric = "SIr",
  source = "real",
  jonckheere_p = jt_real$p.value,
  wilcox_1_2_p = pvals_real[1],
  wilcox_2_3_p = pvals_real[2],
  wilcox_1_3_p = pvals_real[3]
))

# ---- 2. Spatial Information for Polar Maps ----

x_model <- np$load(file.path(BASE_DIR, 'data/sid_model.npy')); x_model <- py_to_r(x_model)
jt_model <- JonckheereTerpstraTest(x_model, g_model, alternative = 'increasing', nperm = 10000)
wilcox_model <- pairwise.wilcox.test(x_model, g_model, p.adjust.method = "BH", alternative = "greater")
pvals_model <- get_wilcox_pvals(wilcox_model)
results <- rbind(results, data.frame(
  metric = "SId",
  source = "model",
  jonckheere_p = jt_model$p.value,
  wilcox_1_2_p = pvals_model[1],
  wilcox_2_3_p = pvals_model[2],
  wilcox_1_3_p = pvals_model[3]
))

x_real <- np$load(file.path(BASE_DIR, 'data/sid_real.npy')); x_real <- py_to_r(x_real)
jt_real <- JonckheereTerpstraTest(x_real, g_real, alternative = 'increasing', nperm = 10000)
wilcox_real <- pairwise.wilcox.test(x_real, g_real, p.adjust.method = "BH", alternative = "greater")
pvals_real <- get_wilcox_pvals(wilcox_real)
results <- rbind(results, data.frame(
  metric = "SId",
  source = "real",
  jonckheere_p = jt_real$p.value,
  wilcox_1_2_p = pvals_real[1],
  wilcox_2_3_p = pvals_real[2],
  wilcox_1_3_p = pvals_real[3]
))

# ---- 3. Resultant Vector Length ----

x_model <- np$load(file.path(BASE_DIR, 'data/rvl_model.npy')); x_model <- py_to_r(x_model)
jt_model <- JonckheereTerpstraTest(x_model, g_model, alternative = 'increasing', nperm = 10000)
wilcox_model <- pairwise.wilcox.test(x_model, g_model, p.adjust.method = "BH", alternative = "greater")
pvals_model <- get_wilcox_pvals(wilcox_model)
results <- rbind(results, data.frame(
  metric = "RVL",
  source = "model",
  jonckheere_p = jt_model$p.value,
  wilcox_1_2_p = pvals_model[1],
  wilcox_2_3_p = pvals_model[2],
  wilcox_1_3_p = pvals_model[3]
))

x_real <- np$load(file.path(BASE_DIR, 'data/rvl_real.npy')); x_real <- py_to_r(x_real)
jt_real <- JonckheereTerpstraTest(x_real, g_real, alternative = 'increasing', nperm = 10000)
wilcox_real <- pairwise.wilcox.test(x_real, g_real, p.adjust.method = "BH", alternative = "greater")
pvals_real <- get_wilcox_pvals(wilcox_real)
results <- rbind(results, data.frame(
  metric = "RVL",
  source = "real",
  jonckheere_p = jt_real$p.value,
  wilcox_1_2_p = pvals_real[1],
  wilcox_2_3_p = pvals_real[2],
  wilcox_1_3_p = pvals_real[3]
))

# --- Percentages (only "real" rows) ---

g_real <- np$load(file.path(BASE_DIR, 'data/g_real_perc.npy')); g_real <- py_to_r(g_real)

# Place cells percentage
x_real <- np$load(file.path(BASE_DIR, 'data/pc_perc_real.npy')); x_real <- py_to_r(x_real)
jt_real <- JonckheereTerpstraTest(x_real, g_real, alternative = 'increasing', nperm = 10000)
wilcox_real <- pairwise.wilcox.test(x_real, g_real, p.adjust.method = "BH", alternative = "greater")
pvals_real <- get_wilcox_pvals(wilcox_real)
results <- rbind(results, data.frame(
  metric = "PlaceCellsPerc",
  source = "real",
  jonckheere_p = jt_real$p.value,
  wilcox_1_2_p = pvals_real[1],
  wilcox_2_3_p = pvals_real[2],
  wilcox_1_3_p = pvals_real[3]
))

# HD cells percentage
x_real <- np$load(file.path(BASE_DIR, 'data/hdc_perc_real.npy')); x_real <- py_to_r(x_real)
jt_real <- JonckheereTerpstraTest(x_real, g_real, alternative = 'increasing', nperm = 10000)
wilcox_real <- pairwise.wilcox.test(x_real, g_real, p.adjust.method = "BH", alternative = "greater")
pvals_real <- get_wilcox_pvals(wilcox_real)
results <- rbind(results, data.frame(
  metric = "HDCellsPerc",
  source = "real",
  jonckheere_p = jt_real$p.value,
  wilcox_1_2_p = pvals_real[1],
  wilcox_2_3_p = pvals_real[2],
  wilcox_1_3_p = pvals_real[3]
))

# Place+HD cells percentage
x_real <- np$load(file.path(BASE_DIR, 'data/phdc_perc_real.npy')); x_real <- py_to_r(x_real)
jt_real <- JonckheereTerpstraTest(x_real, g_real, alternative = 'increasing', nperm = 10000)
wilcox_real <- pairwise.wilcox.test(x_real, g_real, p.adjust.method = "BH", alternative = "greater")
pvals_real <- get_wilcox_pvals(wilcox_real)
results <- rbind(results, data.frame(
  metric = "ConjunctiveCellsPerc",
  source = "real",
  jonckheere_p = jt_real$p.value,
  wilcox_1_2_p = pvals_real[1],
  wilcox_2_3_p = pvals_real[2],
  wilcox_1_3_p = pvals_real[3]
))

# ---- Write to CSV ----

write.csv(results, file = OUTPUT_FILE, row.names = FALSE)
