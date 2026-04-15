# limma_de.R — limma differential expression, called from Python via rpy2.
# Do not run standalone.

run_limma_de <- function(expr_df, groups, contrast = "NEPC-ARPC",
                         weights = NULL, tfx = NULL) {
    library(limma)

    expr_matrix <- as.matrix(expr_df)

    group_factor <- factor(groups, levels = sort(unique(as.character(groups))))

    # Build design matrix: group intercept-free + optional continuous TFx
    if (!is.null(tfx)) {
        tfx_numeric <- as.numeric(tfx)
        design <- model.matrix(~ 0 + group_factor + tfx_numeric)
        colnames(design)[seq_along(levels(group_factor))] <- levels(group_factor)
        colnames(design)[ncol(design)] <- "TFx"
    } else {
        design <- model.matrix(~ 0 + group_factor)
        colnames(design) <- levels(group_factor)
    }

    # Fit with optional observation-level precision weights
    if (!is.null(weights)) {
        w_matrix <- as.matrix(weights)
        fit <- lmFit(expr_matrix, design, weights = w_matrix)
    } else {
        fit <- lmFit(expr_matrix, design)
    }

    cmat <- makeContrasts(contrasts = contrast, levels = design)
    fit2 <- contrasts.fit(fit, cmat)
    fit2 <- eBayes(fit2, trend = TRUE)

    results <- topTable(fit2, number = Inf, sort.by = "none")
    results
}
