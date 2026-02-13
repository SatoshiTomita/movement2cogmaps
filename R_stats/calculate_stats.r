options(warn = -1)

library(DescTools)
library(reticulate)

np <- import("numpy", convert = FALSE)


BASE_DIR <- "/home/USER/vr_to_pc/R_stats"

# Helper function: extract Wilcoxon p-values for pairs 1-2 and 2-3
get_wilcox_pvals <- function(test) {
  # Extract matrix, assume group order as (1, 2, 3)
  mat <- test$p.value
  p_1_2 <- mat[1]
  p_2_3 <- mat[4]
  c(p_1_2, p_2_3)
}

# Initialize results data.frame
results <- data.frame(
  metric = character(),
  source = character(),
  jonckheere_p = numeric(),
  wilcox_1_2_p = numeric(),
  wilcox_2_3_p = numeric(),
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
  wilcox_2_3_p = pvals_model[2]
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
  wilcox_2_3_p = pvals_real[2]
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
  wilcox_2_3_p = pvals_model[2]
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
  wilcox_2_3_p = pvals_real[2]
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
  wilcox_2_3_p = pvals_model[2]
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
  wilcox_2_3_p = pvals_real[2]
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
  wilcox_2_3_p = pvals_real[2]
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
  wilcox_2_3_p = pvals_real[2]
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
  wilcox_2_3_p = pvals_real[2]
))

# ---- Write to CSV ----

write.csv(results, file = file.path(BASE_DIR, "stats_out.csv"), row.names = FALSE)

