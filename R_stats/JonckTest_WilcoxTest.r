# TO RUN FOR GRID CELLS OR RATE OF CHANGE MODELS, REPLACE "data"
# WITH THE RELEVANT FOLDER NAME: "data", "data_roc", or "data_roc_gc"

options(warn=-1)

library(DescTools)
library(reticulate)
np <- import("numpy", convert=FALSE)

g_model <- np$load('data/g_model.npy')
g_model <- py_to_r(g_model)
g_real <- np$load('data/g_real.npy')
g_real <- py_to_r(g_real)

print("####################################################")
print("Spatial Information for Rate Maps")

x_model <- np$load('data/sir_model.npy')
x_model <- py_to_r(x_model)
print("JonckheereTerpstraTest MODEL")
print(JonckheereTerpstraTest(x_model, g_model, alternative='increasing', nperm=10000))
print(pairwise.wilcox.test(x_model, g_model, p.adjust.method="BH", alternative="greater"))

x_real <- np$load('data/sir_real.npy')
x_real <- py_to_r(x_real)
print("JonckheereTerpstraTest EXPERIMENTAL DATA")
print(JonckheereTerpstraTest(x_real, g_real, alternative='increasing', nperm=10000))
print(pairwise.wilcox.test(x_real, g_real, p.adjust.method="BH", alternative="greater"))


print("")
print("####################################################")
print("Spatial Information for Polar Maps")

x_model <- np$load('data/sid_model.npy')
x_model <- py_to_r(x_model)
print("JonckheereTerpstraTest MODEL")
print(JonckheereTerpstraTest(x_model, g_model, alternative='increasing', nperm=10000))
print(pairwise.wilcox.test(x_model, g_model, p.adjust.method="BH", alternative="greater"))

x_real <- np$load('data/sid_real.npy')
x_real <- py_to_r(x_real)
print("JonckheereTerpstraTest EXPERIMENTAL DATA")
print(JonckheereTerpstraTest(x_real, g_real, alternative='increasing', nperm=10000))
print(pairwise.wilcox.test(x_real, g_real, p.adjust.method="BH", alternative="greater"))


print("")
print("####################################################")
print("Resultant Vector Length for Polar Maps")

x_model <- np$load('data/rvl_model.npy')
x_model <- py_to_r(x_model)
print("JonckheereTerpstraTest MODEL")
print(JonckheereTerpstraTest(x_model, g_model, alternative='increasing', nperm=10000))
print(pairwise.wilcox.test(x_model, g_model, p.adjust.method="BH", alternative="greater"))

x_real <- np$load('data/rvl_real.npy')
x_real <- py_to_r(x_real)
print("JonckheereTerpstraTest EXPERIMENTAL DATA")
print(JonckheereTerpstraTest(x_real, g_real, alternative='increasing', nperm=10000))
print(pairwise.wilcox.test(x_real, g_real, p.adjust.method="BH", alternative="greater"))


g_real <- np$load('data/g_real_perc.npy')
g_real <- py_to_r(g_real)

print("####################################################")
print("Place cells percentage")

x_real <- np$load('data/pc_perc_real.npy')
x_real <- py_to_r(x_real)
print("JonckheereTerpstraTest EXPERIMENTAL DATA")
print(JonckheereTerpstraTest(x_real, g_real, alternative='increasing', nperm=10000))
print(pairwise.wilcox.test(x_real, g_real, p.adjust.method="BH", alternative="greater"))

print("####################################################")
print("HD cells percentage")

x_real <- np$load('data/hdc_perc_real.npy')
x_real <- py_to_r(x_real)
print("JonckheereTerpstraTest EXPERIMENTAL DATA")
print(JonckheereTerpstraTest(x_real, g_real, alternative='increasing', nperm=10000))
print(pairwise.wilcox.test(x_real, g_real, p.adjust.method="BH", alternative="greater"))

print("####################################################")
print("Place+HD cells percentage")

x_real <- np$load('data/phdc_perc_real.npy')
x_real <- py_to_r(x_real)
print("JonckheereTerpstraTest EXPERIMENTAL DATA")
print(JonckheereTerpstraTest(x_real, g_real, alternative='increasing', nperm=10000))
print(pairwise.wilcox.test(x_real, g_real, p.adjust.method="BH", alternative="greater"))
