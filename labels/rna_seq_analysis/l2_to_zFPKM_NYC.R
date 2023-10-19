#AUTHOR: Ron Ammar <ron.ammar@bms.com>, John R Thompson <john.thompson@bms.com>
#COMPANY: Bristol-Myers Squibb Co.
#DATE: 2015-07-08
#OBJECTIVE: Perform the zFPKM transform on RNA-seq FPKM data.

## EDITED TO RUN FOR ALREADY LOG2 PROCESSED SAMPLES 

# USAGE
# source("l2_to_zFPKM.R")
# df <- read.csv("path/to/avgl2FPKM.csv", header=TRUE, row.names=1, check.names=FALSE)
# dfZ <- zFPKM(df)
# write.csv(dfZ, "path/to/out.csv", row.names=TRUE)



zFPKM<- function(fpkmDF, assayName="fpkm") {
  return(zFPKMTransform(fpkmDF, assayName)[[2]])
}


removeNanInfRows <- function(fpkm) {
  # Remove FPKM rows containing all NaN values. These are most likely a result
  # of effective lengths = 0 when calculating FPKM.
  return(fpkm[which(!apply(fpkm, 1, function(r) all(is.nan(r) | is.infinite(r)))), ])
}


zFPKMTransform <- function(fpkmDF, assayName) {
  # Helper function for zFPKM output and plotting. Do not call directly.
  #
  # Args:
  #   Defined by zFPKM() and zFPKMPlot()
  #
  # Returns:
  #   Internal objects used for zFPKM calculations and plotting.

  if (is(fpkmDF, "SummarizedExperiment")) {
    fpkmDF <- assay(fpkmDF, assayName)
  }

  fpkmDF <- removeNanInfRows(fpkmDF)

  zFPKMDF <- data.frame(row.names=row.names(fpkmDF))
  outputs <- list()
  for (c in colnames(fpkmDF)) {
    output <- zFPKMCalc(fpkmDF[, c])
    zFPKMDF[, c] <- output[["z"]]
    outputs[[c]] <- output
  }

  return(list(outputs, zFPKMDF))
}


zFPKMCalc <- function(fpkm) {
  # Performs the zFPKM transform on RNA-seq FPKM data. This involves fitting a
  # Gaussian distribution based on the right side of the FPKM distribution, as
  # described by Hart et al., 2013 (Pubmed ID 24215113). The zFPKM transformed
  # FPKM values represent normalized FPKM data, and, according to Hart et al.,
  # a zFPKM value >= -3 can be considered expressed (has an active promoter).
  #
  # Args:
  #   fpkm: a vector of raw FPKM values. NOTE: these are NOT log_2 transformed.
  #
  # Returns:
  #   The zFPKM transformed vector of the input FPKM data

  if (!is.numeric(fpkm)) {
    stop("argument 'fpkm' must be numeric")
  }

  # AKIRA NAIR: 
  # NOTE: assuming values are already in log 2 format
  fpkmLog2 <- fpkm

  # Compute kernel density estimate
  d <- density(fpkmLog2)

  # Set the maximum point in the density as the mean for the fitted Gaussian
  mu <- d[["x"]][which.max(d[["y"]])]

  # Determine the standard deviation
  U <- mean(fpkmLog2[fpkmLog2 > mu])
  stdev <- (U - mu) * sqrt(pi / 2)

  # Compute zFPKM transform
  zFPKM <- (fpkmLog2 - mu) / stdev

  result <- ZFPKMResult(zFPKM, d, mu, stdev)

  return(result)
}

ZFPKMResult <- function(zfpkmVector, density, mu, stdev) {
  # S3 class to store zFPKM vector and related metrics to be used in plotting

  zfpkmRes <- list(
    z = zfpkmVector,
    d = density,
    m = mu,
    s = stdev
  )

  class(zfpkmRes) <- append(class(zfpkmRes), "zFPKM")
  return(zfpkmRes)
}