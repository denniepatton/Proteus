# limma_de.R — limma differential expression, called from Python via rpy2.
# Do not run standalone.

run_limma_de <- function(expr_df, groups, contrast = "NEPC-ARPC", tfx = NULL) {
    library(limma)

    expr_matrix <- as.matrix(expr_df)

    group_factor <- factor(groups, levels = sort(unique(as.character(groups))))

    # Build design matrix: group intercept-free + optional continuous TFx
    if (!is.null(tfx)) {
        design <- model.matrix(~ 0 + group_factor + as.numeric(tfx))
        colnames(design)[seq_along(levels(group_factor))] <- levels(group_factor)
        colnames(design)[ncol(design)] <- "TFx"
    } else {
        design <- model.matrix(~ 0 + group_factor)
        colnames(design) <- levels(group_factor)
    }

    fit <- limma::lmFit(expr_matrix, design)

    cmat <- limma::makeContrasts(contrasts = contrast, levels = design)
    fit2 <- limma::contrasts.fit(fit, cmat)
    fit2 <- limma::eBayes(fit2, trend = TRUE)

    results <- limma::topTable(fit2, number = Inf, sort.by = "none")
    se <- fit2$stdev.unscaled[, 1] * sqrt(fit2$s2.post)
    results$standard_error <- se[rownames(results)]
    results$df_total <- fit2$df.total[rownames(results)]
    results
}
