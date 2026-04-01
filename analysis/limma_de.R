# limma_de.R — limma differential expression, called from Python via rpy2.
# Do not run standalone.

run_limma_de <- function(expr_df, groups, contrast = "NEPC-ARPC") {
    library(limma)

    expr_matrix <- as.matrix(expr_df)

    group_factor <- factor(groups, levels = sort(unique(as.character(groups))))
    design <- model.matrix(~ 0 + group_factor)
    colnames(design) <- levels(group_factor)

    fit  <- lmFit(expr_matrix, design)
    cmat <- makeContrasts(contrasts = contrast, levels = design)
    fit2 <- contrasts.fit(fit, cmat)
    fit2 <- eBayes(fit2, trend = TRUE)

    results <- topTable(fit2, number = Inf, sort.by = "none")
    results
}
