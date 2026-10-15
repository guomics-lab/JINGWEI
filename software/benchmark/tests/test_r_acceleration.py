from pathlib import Path
import subprocess

import pytest

from jingwei_benchmark.methods import availability, find_rscript, r_environment


def test_mice_acceleration_preserves_native_chains_rng_and_namespace(tmp_path):
    if not availability(names=["mice-pmm"], mice_backend="r")["mice-pmm"]["available"]:
        pytest.skip("MICE unavailable")
    helper = Path(__file__).parents[1] / "src/jingwei_benchmark/r_acceleration.R"
    code = tmp_path / "equivalence.R"
    code.write_text("""
source(commandArgs(trailingOnly=TRUE)[1])
original <- get('remove.lindep', asNamespace('mice'))
for (wide in c(FALSE, TRUE)) {
  set.seed(13)
  values <- matrix(rnorm(if(wide) 480 else 192), nrow=if(wide) 16 else 24)
  values[sample(length(values), floor(length(values)*.22))] <- NA_real_
  # Exercise constants/duplicate columns and native logged-event handling.
  values <- cbind(values, constant=1, duplicate=values[,1],
                  near_duplicate=values[,1]+rnorm(nrow(values),sd=1e-6), all_missing=NA_real_)
  colnames(values) <- paste0('V', seq_len(ncol(values)))
  for (method in c('norm','cart','pmm')) {
    run <- function() mice::mice(as.data.frame(values), m=2, maxit=2,
                                 method=method, seed=47, printFlag=FALSE)
    native <- suppressWarnings(with_mice_acceleration(run, enabled=FALSE))
    native_seed <- .Random.seed
    fast <- suppressWarnings(with_mice_acceleration(run, enabled=TRUE))
    stopifnot(fast$acceleration$applied, identical(.Random.seed,native_seed))
    for (field in c('imp','predictorMatrix','loggedEvents','chainMean','chainVar','iteration','lastSeedValue')) {
      stopifnot(identical(native$value[[field]],fast$value[[field]]))
    }
    stopifnot(identical(get('remove.lindep',asNamespace('mice')),original))
  }
}
# Restoration on an exception and fallback when upstream has changed.
try(with_mice_acceleration(function() stop('deliberate')),silent=TRUE)
stopifnot(identical(get('remove.lindep',asNamespace('mice')),original))
changed <- original; body(changed) <- quote(stop('changed implementation'))
utils::assignInNamespace('remove.lindep', changed, ns='mice')
fallback <- with_mice_acceleration(function() 42)
stopifnot(!fallback$acceleration$applied, fallback$value==42)
utils::assignInNamespace('remove.lindep', original, ns='mice')
cat('MICE equivalence, RNG, restoration and fallback passed\\n')
""", encoding="utf-8")
    result = subprocess.run([find_rscript(), "--vanilla", str(code), str(helper.resolve())],
                            env=r_environment(), capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "equivalence" in result.stdout
