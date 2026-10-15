# Execution-only changes to official R algorithms. No package files are edited.
# Loaded in an isolated method process. Always restore the namespace binding.
with_mice_acceleration <- function(run, enabled=TRUE) {
  status <- list(requested=enabled, applied=FALSE, reason='native execution')
  if (!enabled) return(list(value=run(), acceleration=status))
  namespace <- asNamespace('mice')
  original <- get('remove.lindep', envir=namespace, inherits=FALSE)
  # Only a principal submatrix of cor(..., use="all.obs") is eligible. Guard
  # the complete upstream body so a future implementation falls back safely.
  expected <- quote({
    if (ncol(x) == 0) return(NULL)
    if (eps == 0) return(rep.int(TRUE, ncol(x)))
    if (eps < 0) stop("\n Argument 'eps' must be positive.")
    if (allow.na && sum(ry) == 0) return(rep.int(TRUE, ncol(x)))
    xobs <- x[ry, , drop = FALSE]
    yobs <- as.numeric(y[ry])
    if (var(yobs) < eps) return(rep(FALSE, ncol(xobs)))
    keep <- unlist(apply(xobs, 2, var) > eps)
    keep[is.na(keep)] <- FALSE
    highcor <- suppressWarnings(unlist(apply(xobs, 2, cor, yobs) < maxcor))
    keep <- keep & highcor
    if (all(!keep)) updateLog(out = "All predictors are constant or have too high correlation.", frame = frame)
    k <- sum(keep)
    if (k <= 1L) return(keep)
    cx <- cor(xobs[, keep, drop = FALSE], use = "all.obs")
    eig <- eigen(cx, symmetric = TRUE)
    ncx <- cx
    while (eig$values[k]/eig$values[1] < eps) {
      j <- seq_len(k)[order(abs(eig$vectors[, k]), decreasing = TRUE)[1]]
      keep[keep][j] <- FALSE
      ncx <- cx[keep[keep], keep[keep], drop = FALSE]
      k <- k - 1
      eig <- eigen(ncx)
    }
    if (!all(keep)) {
      out <- paste(dimnames(x)[[2]][!keep], collapse = ", ")
      updateLog(out = out, frame = frame)
    }
    return(keep)
  })
  # Ignore redundant braces around single statements, but no expressions or
  # arguments. This covers upstream's formatting without weakening the guard.
  normalize <- function(node) {
    if (!is.call(node)) return(node)
    parts <- lapply(as.list(node), normalize)
    if (identical(parts[[1]], as.name('{')) && length(parts)==2L) return(parts[[2]])
    as.call(parts)
  }
  if (!identical(normalize(body(original)), normalize(expected))) {
    status$reason <- 'unsupported mice::remove.lindep body; using native execution'
    return(list(value=run(), acceleration=status))
  }
  replacements <- 0L
  replace_eigen <- function(node) {
    if (identical(node, quote(eig <- eigen(cx, symmetric=TRUE)))) {
      return(quote(eig <- initial_eigen <- eigen(cx, symmetric=TRUE)))
    }
    if (identical(node, quote(eigen(ncx)))) {
      replacements <<- replacements + 1L
      # Upstream can produce the same correlation submatrix on successive
      # iterations. Reuse only when all values, dimensions and names match.
      # This preserves its predictor decisions, including its indexing rules.
      return(quote(if (identical(ncx, cx)) initial_eigen else eigen(ncx, symmetric=TRUE)))
    }
    if (is.call(node)) return(as.call(lapply(as.list(node), replace_eigen)))
    node
  }
  optimized <- original
  body(optimized) <- replace_eigen(body(original))
  if (replacements != 1L) stop('invalid MICE acceleration replacement count')
  # Recompile the edited closure, retaining its original namespace environment.
  optimized <- compiler::cmpfun(optimized)
  utils::assignInNamespace('remove.lindep', optimized, ns='mice')
  on.exit(utils::assignInNamespace('remove.lindep', original, ns='mice'), add=TRUE)
  status$applied <- TRUE
  status$reason <- 'reuse identical correlation eigendecompositions; explicit symmetric solver'
  list(value=run(), acceleration=status)
}
