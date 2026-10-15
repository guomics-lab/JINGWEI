# Fill missing R backends; Linux/macOS run with conda R and conda compilers.
args <- commandArgs(trailingOnly = TRUE)
root <- normalizePath(args[1], mustWork = TRUE)
library <- if (length(args) >= 2) args[2] else if (.Platform$OS.type == "windows") {
    legacy <- file.path(root, ".r-library")
    if (file.exists(file.path(legacy, "jsonlite", "DESCRIPTION"))) legacy else file.path(legacy, "windows")
} else .libPaths()[1]
dir.create(library, recursive = TRUE, showWarnings = FALSE)
.libPaths(c(normalizePath(library), .libPaths()))
options(repos = c(CRAN = "https://cloud.r-project.org"), timeout = 300)
cran <- c("jsonlite", "mice", "missForest", "norm", "rrcovNA", "VIM",
          "glmnet", "imputeLCMD", "ggplot2", "reshape2")
missing <- cran[!vapply(cran, requireNamespace, logical(1), quietly = TRUE)]
if (length(missing)) install.packages(missing, lib = library)
bio <- c("Biobase", "impute", "pcaMethods")
missing <- bio[!vapply(bio, requireNamespace, logical(1), quietly = TRUE)]
if (length(missing)) {
    if (!requireNamespace("BiocManager", quietly = TRUE)) install.packages("BiocManager", lib = library)
    BiocManager::install(missing, lib = library, ask = FALSE, update = FALSE)
}
for (pkg in c("SeqKnn", "GMSimpute")) {
    if (!requireNamespace(pkg, quietly = TRUE)) {
        archive <- list.files(file.path(root, ".method-sources", "NAguideR"),
                              pattern = paste0("^", pkg, "_.*\\.tar\\.gz$"), full.names = TRUE)
        if (length(archive) != 1) stop("Missing original NAguideR archive: ", pkg)
        install.packages(archive, repos = NULL, type = "source", lib = library)
    }
}
all <- c(cran, bio, "SeqKnn", "GMSimpute")
missing <- all[!vapply(all, requireNamespace, logical(1), quietly = TRUE)]
if (length(missing)) stop("R installation incomplete: ", paste(missing, collapse = ", "))
cat("All R backends installed in ", library, "\n")
