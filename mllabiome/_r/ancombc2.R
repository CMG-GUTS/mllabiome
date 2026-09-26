args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 18) stop("Internal error: unexpected number of ANCOM-BC2 arguments")

suppressPackageStartupMessages(library(ANCOMBC))

counts_path <- args[[1]]
aggregate_path <- args[[2]]
metadata_path <- args[[3]]
levels_path <- args[[4]]
output_dir <- args[[5]]
fix_formula <- args[[6]]
rand_formula <- if (args[[7]] == "__NONE__") NULL else args[[7]]
group <- if (args[[8]] == "__NONE__") NULL else args[[8]]
p_adjust_method <- args[[9]]
pseudo_sens <- as.integer(args[[10]]) != 0
prv_cut <- as.numeric(args[[11]])
alpha <- as.numeric(args[[12]])
struc_zero <- as.integer(args[[13]]) != 0
neg_lb <- as.integer(args[[14]]) != 0
global_test <- as.integer(args[[15]]) != 0
pairwise <- as.integer(args[[16]]) != 0
workers <- as.integer(args[[17]])
seed <- as.integer(args[[18]])

counts <- read.delim(counts_path, row.names = 1, check.names = FALSE, quote = "", comment.char = "")
counts <- as.matrix(counts)
storage.mode(counts) <- "double"
aggregate <- read.delim(aggregate_path, row.names = 1, check.names = FALSE, quote = "", comment.char = "")
aggregate <- as.matrix(aggregate)
storage.mode(aggregate) <- "double"
meta <- read.delim(metadata_path, row.names = 1, check.names = FALSE, quote = "", comment.char = "")
levels <- readLines(levels_path, warn = FALSE)
if (length(levels) > 0 && "mll_target" %in% colnames(meta)) meta$mll_target <- factor(meta$mll_target, levels = levels)
if ("mll_cluster" %in% colnames(meta)) meta$mll_cluster <- factor(meta$mll_cluster)
if (!identical(colnames(counts), rownames(meta))) stop("Count samples and metadata samples do not match")

set.seed(seed)
out <- ancombc2(
    data = counts,
    taxa_are_rows = TRUE,
    aggregate_data = aggregate,
    meta_data = meta,
    fix_formula = fix_formula,
    rand_formula = rand_formula,
    p_adj_method = p_adjust_method,
    pseudo = 0,
    pseudo_sens = pseudo_sens,
    prv_cut = prv_cut,
    lib_cut = 0,
    s0_perc = 0.05,
    group = group,
    struc_zero = struc_zero,
    neg_lb = neg_lb,
    alpha = alpha,
    n_cl = workers,
    verbose = FALSE,
    global = global_test,
    pairwise = pairwise,
    dunnet = FALSE,
    trend = FALSE,
    iter_control = list(tol = 0.01, max_iter = 20, verbose = FALSE),
    em_control = list(tol = 1e-05, max_iter = 100),
    lme_control = lme4::lmerControl(),
    mdfdr_control = list(fwer_ctrl_method = "holm", B = 100),
    trend_control = NULL
)

dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
write_tab <- function(value, name) {
    if (is.null(value)) return(invisible(NULL))
    tab <- as.data.frame(value, check.names = FALSE)
    if (nrow(tab) == 0) return(invisible(NULL))
    write.table(tab, file.path(output_dir, name), sep = "\t", row.names = FALSE, quote = FALSE, na = "")
}
write_tab(out$res, "primary.tsv")
write_tab(out$res_global, "global.tsv")
write_tab(out$res_pair, "pairwise.tsv")
write_tab(out$zero_ind, "structural_zeros.tsv")
write_tab(out$ss_tab, "sensitivity.tsv")
