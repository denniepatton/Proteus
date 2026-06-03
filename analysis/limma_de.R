# limma_de.R — limma differential expression, called from Python via rpy2.
# Do not run standalone.

run_limma_de <- function(expr_df, groups, contrast = "NEPC-ARPC", weights = NULL) {
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

    # Predicted-expression limma uses subtype-only contrasts. Observation-level
    # Proteus precision weights already encode uncertainty differences that track
    # purity / TFx, so adding TFx as an explicit covariate would double-count the
    # same information and break the shared weighting contract across cohorts.
    design <- model.matrix(~ 0 + group_factor)
    colnames(design)[seq_along(levels(group_factor))] <- levels(group_factor)

    fit <- limma::lmFit(expr_matrix, design, weights = weights_matrix)

    cmat <- limma::makeContrasts(contrasts = contrast, levels = design)
    fit2 <- limma::contrasts.fit(fit, cmat)
    # Use limma-trend with robust empirical Bayes exactly as requested. We keep
    # the weighted lmFit front-end but do not invoke voom() or treat().
    fit2 <- limma::eBayes(fit2, trend = TRUE, robust = TRUE)

    results <- limma::topTable(
        fit2,
        coef = colnames(cmat)[1],
        number = Inf,
        adjust.method = "BH",
        sort.by = "none"
    )
    se <- fit2$stdev.unscaled[, 1] * sqrt(fit2$s2.post)
    results$standard_error <- se[rownames(results)]
    results$df_total <- fit2$df.total[rownames(results)]
    results
}
