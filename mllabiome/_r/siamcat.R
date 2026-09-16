args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 1) stop("Usage: siamcat.R <fit|predict> ...")

suppressPackageStartupMessages(library(SIAMCAT))

read_feature_matrix <- function(path) {
    tab <- read.delim(
        path,
        row.names = 1,
        check.names = FALSE,
        quote = "",
        comment.char = ""
    )
    feat <- t(as.matrix(tab))
    storage.mode(feat) <- "double"
    feat
}

as_flag <- function(x) {
    as.integer(x) != 0
}

mode <- args[[1]]

if (mode == "fit") {
    if (length(args) != 19) {
        stop("Internal error: unexpected number of SIAMCAT fit arguments")
    }

    x_path <- args[[2]]
    y_path <- args[[3]]
    model_path <- args[[4]]
    method <- args[[5]]
    filter_method <- args[[6]]
    filter_cutoff <- as.numeric(args[[7]])
    norm_method <- args[[8]]
    num_folds <- as.integer(args[[9]])
    num_resample <- as.integer(args[[10]])
    measure <- args[[11]]
    grid_size <- as.integer(args[[12]])
    min_nonzero <- as.integer(args[[13]])
    perform_fs <- as_flag(args[[14]])
    no_features <- as.integer(args[[15]])
    fs_method <- args[[16]]
    fs_direction <- args[[17]]
    seed <- as.integer(args[[18]])
    verbose <- as.integer(args[[19]])

    set.seed(seed)

    feat <- read_feature_matrix(x_path)

    labels <- read.delim(
        y_path,
        row.names = 1,
        check.names = FALSE,
        quote = "",
        comment.char = ""
    )
    if (!"label" %in% colnames(labels)) {
        stop("Label file must contain a 'label' column")
    }

    if (!all(colnames(feat) %in% rownames(labels))) {
        stop("Feature samples and label samples do not match")
    }

    labels <- labels[colnames(feat), "label", drop = TRUE]

    
    
    
    group <- ifelse(as.integer(labels) == 1L, "case", "control")
    names(group) <- colnames(feat)

    sc <- siamcat(
        feat = feat,
        label = group,
        case = "case",
        verbose = verbose
    )

    sc <- filter.features(
        sc,
        filter.method = filter_method,
        cutoff = filter_cutoff,
        feature.type = "original",
        verbose = verbose
    )

    sc <- normalize.features(
        sc,
        norm.method = norm_method,
        feature.type = "filtered",
        verbose = verbose
    )

    sc <- create.data.split(
        sc,
        num.folds = num_folds,
        num.resample = num_resample,
        stratify = TRUE,
        verbose = verbose
    )

    sc <- train.model(
        sc,
        method = method,
        measure = measure,
        grid.size = grid_size,
        min.nonzero = min_nonzero,
        perform.fs = perform_fs,
        param.fs = list(
            no_features = no_features,
            method = fs_method,
            direction = fs_direction
        ),
        feature.type = "normalized",
        verbose = verbose
    )

    saveRDS(sc, model_path)

} else if (mode == "predict") {
    if (length(args) != 5) {
        stop("Internal error: unexpected number of SIAMCAT predict arguments")
    }

    model_path <- args[[2]]
    x_path <- args[[3]]
    output_path <- args[[4]]
    verbose <- as.integer(args[[5]])

    trained <- readRDS(model_path)
    feat <- read_feature_matrix(x_path)

    holdout <- siamcat(
        feat = feat,
        verbose = verbose
    )

    holdout <- make.predictions(
        siamcat = trained,
        siamcat.holdout = holdout,
        normalize.holdout = TRUE,
        verbose = verbose
    )

    pred <- pred_matrix(holdout, verbose = 0)
    if (is.null(pred)) {
        stop("SIAMCAT returned no prediction matrix")
    }

    if (is.null(dim(pred))) {
        score <- as.numeric(pred)
        names(score) <- names(pred)
    } else {
        score <- rowMeans(pred, na.rm = TRUE)
        names(score) <- rownames(pred)
    }

    if (any(!is.finite(score))) {
        stop("SIAMCAT returned non-finite prediction scores")
    }

    out <- data.frame(
        sample_id = names(score),
        score = as.numeric(score),
        check.names = FALSE
    )

    write.table(
        out,
        file = output_path,
        sep = "\t",
        quote = FALSE,
        row.names = FALSE
    )

} else {
    stop(paste0("Unknown SIAMCAT runner mode: ", mode))
}
