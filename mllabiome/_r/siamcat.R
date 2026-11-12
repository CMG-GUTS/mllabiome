args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 1) stop("Usage: siamcat.R <fit|predict> ...")

siamcat_progress <- function(phase) {
    message(sprintf("[SIAMCAT:R] %s | %s", format(Sys.time(), "%H:%M:%S"), phase))
}

siamcat_progress("Loading SIAMCAT library")
suppressPackageStartupMessages(library(SIAMCAT))
siamcat_progress("SIAMCAT library loaded")

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
    if (length(args) != 21) {
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
    groups_path <- args[[20]]
    split_path <- args[[21]]

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

    meta_df <- NULL
    inseparable_var <- NULL
    if (groups_path != "__NONE__") {
        groups <- read.delim(
            groups_path,
            row.names = 1,
            check.names = FALSE,
            quote = "",
            comment.char = ""
        )
        if (!"cv_group" %in% colnames(groups)) {
            stop("Group file must contain a 'cv_group' column")
        }
        if (!all(colnames(feat) %in% rownames(groups))) {
            stop("Feature samples and group samples do not match")
        }
        groups <- groups[colnames(feat), "cv_group", drop = TRUE]
        if (any(is.na(groups))) {
            stop("Group file contains missing values")
        }
        meta_df <- data.frame(
            cv_group = as.character(groups),
            row.names = colnames(feat),
            check.names = FALSE
        )
        inseparable_var <- "cv_group"
    }

    siamcat_progress(sprintf("Building training object (%d samples, %d features)", ncol(feat), nrow(feat)))
    sc <- siamcat(
        feat = feat,
        label = group,
        case = "case",
        meta = meta_df,
        verbose = verbose
    )

    siamcat_progress("Filtering features")
    sc <- filter.features(
        sc,
        filter.method = filter_method,
        cutoff = filter_cutoff,
        feature.type = "original",
        verbose = verbose
    )

    siamcat_progress("Normalizing features")
    sc <- normalize.features(
        sc,
        norm.method = norm_method,
        feature.type = "filtered",
        verbose = verbose
    )

    siamcat_progress("Loading validated, class-balanced internal resampling folds")
    split_plan <- read.delim(
        split_path,
        row.names = 1,
        check.names = FALSE,
        quote = "",
        comment.char = ""
    )
    ids <- colnames(feat)
    if (!setequal(rownames(split_plan), ids) || ncol(split_plan) != num_resample) {
        stop("SIAMCAT internal split plan has missing samples or the wrong number of repetitions")
    }
    split_plan <- split_plan[ids, , drop = FALSE]
    training <- vector("list", num_resample)
    testing <- vector("list", num_resample)
    group_values <- if (is.null(meta_df)) NULL else as.character(meta_df[ids, "cv_group"])
    for (r in seq_len(num_resample)) {
        allocation <- as.integer(split_plan[[r]])
        if (anyNA(allocation) || !setequal(unique(allocation), seq_len(num_folds))) {
            stop("SIAMCAT internal split plan does not assign every fold")
        }
        training[[r]] <- vector("list", num_folds)
        testing[[r]] <- vector("list", num_folds)
        for (f in seq_len(num_folds)) {
            is_test <- allocation == f
            is_train <- !is_test
            for (klass in c(0L, 1L)) {
                if (sum(as.integer(labels[is_train]) == klass) < 3L ||
                    sum(as.integer(labels[is_test]) == klass) < 1L) {
                    stop(sprintf("SIAMCAT unsafe internal CV: repetition %d fold %d class %d has %d training and %d test observations", r, f, klass, sum(as.integer(labels[is_train]) == klass), sum(as.integer(labels[is_test]) == klass)))
                }
            }
            if (!is.null(group_values) &&
                length(intersect(unique(group_values[is_train]), unique(group_values[is_test]))) > 0L) {
                stop("SIAMCAT internal CV violates the group-disjoint requirement")
            }
            training[[r]][[f]] <- ids[is_train]
            testing[[r]][[f]] <- ids[is_test]
        }
    }
    data_split(sc) <- list(
        training.folds = training,
        test.folds = testing,
        num.resample = num_resample,
        num.folds = num_folds
    )

    siamcat_progress("Training SIAMCAT model")
    max_attempts <- if (method %in% c("lasso", "enet", "ridge")) 8L else 1L
    for (attempt in seq_len(max_attempts)) {
        set.seed(as.integer((seed + 1009 * (attempt - 1L)) %% .Machine$integer.max))
        fitted <- tryCatch(
            train.model(
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
            ),
            error = function(e) e
        )
        if (!inherits(fitted, "error")) {
            sc <- fitted
            break
        }
        msg <- conditionMessage(fitted)
        if (attempt == max_attempts ||
            !grepl("one multinomial or binomial class has 1 or 0 observations", msg, fixed = TRUE)) {
            stop(fitted)
        }
        siamcat_progress(sprintf("Retrying glmnet internal CV with another seed (%d/%d)", attempt + 1L, max_attempts))
    }

    siamcat_progress("Saving trained SIAMCAT model")
    saveRDS(sc, model_path)
    siamcat_progress("Fit completed")

} else if (mode == "predict") {
    if (length(args) != 5) {
        stop("Internal error: unexpected number of SIAMCAT predict arguments")
    }

    model_path <- args[[2]]
    x_path <- args[[3]]
    output_path <- args[[4]]
    verbose <- as.integer(args[[5]])

    siamcat_progress("Loading trained model for holdout prediction")
    trained <- readRDS(model_path)
    feat <- read_feature_matrix(x_path)

    holdout <- siamcat(
        feat = feat,
        verbose = verbose
    )

    siamcat_progress(sprintf("Predicting on %d holdout samples", ncol(feat)))
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

    siamcat_progress("Writing holdout predictions")
    write.table(
        out,
        file = output_path,
        sep = "\t",
        quote = FALSE,
        row.names = FALSE
    )
    siamcat_progress("Prediction completed")

} else {
    stop(paste0("Unknown SIAMCAT runner mode: ", mode))
}
