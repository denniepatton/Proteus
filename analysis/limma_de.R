# limma_de.R — limma differential expression, called from Python via rpy2.
# Do not run standalone.

run_limma_de <- function(expr_df, groups, contrast = "NEPC-ARPC", tfx, weights = NULL) {
    library(limma)

    expr_matrix <- as.matrix(expr_df)
    weights_matrix <- NULL
    if (!is.null(weights)) {
        weights_matrix <- as.matrix(weights)
        if (!all(dim(weights_matrix) == dim(expr_matrix))) {
            stop("weights matrix dimensions must match expression matrix")
        }
    }

    group_factor <- factor(groups, levels = sort(unique(as.character(groups))))

    if (missing(tfx) || is.null(tfx)) {
        design <- model.matrix(~ 0 + group_factor)
        colnames(design)[seq_along(levels(group_factor))] <- levels(group_factor)
    } else {
        if (length(tfx) != ncol(expr_matrix)) {
            stop("TFx covariate length must match expression columns")
        }
        design <- model.matrix(~ 0 + group_factor + as.numeric(tfx))
        colnames(design)[seq_along(levels(group_factor))] <- levels(group_factor)
        colnames(design)[ncol(design)] <- "TFx"
    }

    fit <- limma::lmFit(expr_matrix, design, weights = weights_matrix)

    cmat <- limma::makeContrasts(contrasts = contrast, levels = design)
    fit2 <- limma::contrasts.fit(fit, cmat)
    fit2 <- limma::eBayes(fit2, trend = TRUE)

    results <- limma::topTable(fit2, number = Inf, sort.by = "none")
    se <- fit2$stdev.unscaled[, 1] * sqrt(fit2$s2.post)
    results$standard_error <- se[rownames(results)]
    results$df_total <- fit2$df.total[rownames(results)]
    results
}
