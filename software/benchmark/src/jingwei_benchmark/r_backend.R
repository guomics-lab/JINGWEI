# Official R package adapters; input and output are samples x features.
args <- commandArgs(trailingOnly=TRUE)
if (length(args) >= 1 && args[1] == '--check') {
  base_pkgs <- c('R', 'base', 'compiler', 'datasets', 'graphics', 'grDevices',
                 'grid', 'methods', 'parallel', 'splines', 'stats', 'stats4',
                 'tcltk', 'tools', 'utils')
  avail_map <- new.env(parent=emptyenv())
  ver_map <- new.env(parent=emptyenv())
  visiting <- new.env(parent=emptyenv())
  parse_deps <- function(raw) {
    if (is.null(raw) || is.na(raw) || !nzchar(raw)) return(character(0))
    parts <- trimws(strsplit(raw, ',', fixed=TRUE)[[1]])
    parts <- sub('[[:space:]]*\\(.*$', '', parts)
    parts[nzchar(parts) & !(parts %in% base_pkgs)]
  }
  has_pkg <- function(pkg) {
    if (pkg %in% base_pkgs) return(TRUE)
    if (exists(pkg, envir=avail_map, inherits=FALSE)) return(get(pkg, envir=avail_map))
    if (exists(pkg, envir=visiting, inherits=FALSE)) return(TRUE)
    desc <- suppressWarnings(utils::packageDescription(pkg, fields=c('Version', 'Depends', 'Imports')))
    if (!is.list(desc) || is.na(desc[['Version']])) {
      assign(pkg, FALSE, envir=avail_map)
      return(FALSE)
    }
    assign(pkg, TRUE, envir=visiting)
    on.exit(if (exists(pkg, envir=visiting, inherits=FALSE)) rm(list=pkg, envir=visiting), add=TRUE)
    deps <- unique(c(parse_deps(desc[['Depends']]), parse_deps(desc[['Imports']])))
    ok <- all(vapply(deps, has_pkg, logical(1)))
    assign(pkg, ok, envir=avail_map)
    if (ok) assign(pkg, as.character(desc[['Version']]), envir=ver_map)
    ok
  }
  dll_ok <- suppressWarnings(requireNamespace('jsonlite', quietly=TRUE))
  for (pkg in args[-1]) {
    present <- dll_ok && has_pkg(pkg)
    ver <- if (present) get(pkg, envir=ver_map) else ''
    cat(pkg, if (present) 'TRUE' else 'FALSE', ver, sep='\t'); cat('\n')
  }
  quit(status=0)
}
tryCatch({
  if (length(args) < 6) stop('expected method input output seed parameters source_dir')
  method <- args[1]
  if (!requireNamespace('jsonlite',quietly=TRUE)) stop('missing R package: jsonlite')
  pars <- jsonlite::fromJSON(args[5])
  binary <- length(args)>=8 && args[8]=='binary'
  if (binary) {
    shape <- as.integer(pars[['matrix_shape']])
    if (length(shape)!=2 || anyNA(shape) || any(shape<=0) || file.info(args[2])$size!=prod(shape)*8) stop('invalid binary matrix shape or size')
    connection <- file(args[2], 'rb')
    values <- tryCatch(readBin(connection, 'double', n=prod(shape), size=8, endian='little'), finally=close(connection))
    input <- matrix(values, nrow=shape[1], ncol=shape[2], byrow=TRUE,
                    dimnames=list(as.character(seq_len(shape[1])),as.character(seq_len(shape[2])-1L)))
    input[is.nan(input)] <- NA_real_
  } else {
    input <- as.matrix(utils::read.csv(args[2], check.names=FALSE))
  }
  storage.mode(input) <- 'double'
  param <- function(name, default) if (is.null(pars[[name]])) default else pars[[name]]
  need <- function(pkg) if (!requireNamespace(pkg,quietly=TRUE)) stop(paste('missing R package:',pkg))
  seed <- as.integer(args[4]); set.seed(seed)
  acceleration <- list(applied=FALSE,reason='native algorithm')
  execution <- list()
  x <- t(input)
  k <- as.integer(param('k',10))
  result <- switch(method,
    'bpca' =, 'svd' = {
      need('pcaMethods')
      rank <- max(1L,min(as.integer(param('rank',10)),nrow(x)-1L,ncol(x)-1L))
      fit <- pcaMethods::pca(x,nPcs=rank,method=if (method=='bpca') 'bpca' else 'svdImpute',maxSteps=100)
      t(pcaMethods::completeObs(fit))
    },
    'naguider-knn' = {
      need('impute'); t(impute::impute.knn(x,k=min(k,nrow(x)-1L),rowmax=1,colmax=1)$data)
    },
    'seqknn' = {
      need('SeqKnn'); t(SeqKnn::SeqKNN(as.data.frame(x),k=min(k,nrow(x)-1L)))
    },
    'lls' = {
      need('pcaMethods'); t(t(pcaMethods::completeObs(pcaMethods::llsImpute(t(x),k=min(k,nrow(x)-1L)))))
    },
    'mle' = {
      need('norm'); stats <- norm::prelim.norm(x); theta <- norm::em.norm(stats,showits=FALSE)
      norm::rngseed(seed); t(norm::imp.norm(stats,theta,x))
    },
    'qrilc' = {
      need('imputeLCMD'); t(imputeLCMD::impute.QRILC(x,tune.sigma=param('tune_sigma',1))[[1]])
    },
    'mindet' = {
      need('imputeLCMD'); t(imputeLCMD::impute.MinDet(x,q=param('q',.01)))
    },
    'minprob' = {
      need('imputeLCMD'); t(imputeLCMD::impute.MinProb(x,q=param('q',.01),tune.sigma=param('tune_sigma',1)))
    },
    'impseq' = {
      need('rrcovNA'); t(rrcovNA::impSeq(as.data.frame(x)))
    },
    'impseqrob' = {
      need('rrcovNA'); t(rrcovNA::impSeqRob(as.data.frame(x),alpha=param('alpha',.9))$x)
    },
    'irm' = {
      need('VIM'); t(as.matrix(VIM::irmi(as.data.frame(x),trace=FALSE,imp_var=FALSE)))
    },
    'mice-norm' =, 'mice-cart' =, 'mice-pmm' = {
      need('mice'); m <- as.integer(param('m',5))
      values <- if (method=='mice-pmm') input else x
      # Resolve relative to this adapter, including frozen/relocated runtimes.
      helper <- file.path(dirname(normalizePath(sub('^--file=', '', grep('^--file=', commandArgs(), value=TRUE)[1]))), 'r_acceleration.R')
      if (!file.exists(helper)) stop('missing bundled R acceleration adapter')
      sys.source(helper, envir=environment())
      fitted <- with_mice_acceleration(function() {
        options <- list(data=as.data.frame(values),m=m,maxit=param('max_iter',5),seed=seed,
                        method=sub('mice-','',method,fixed=TRUE),printFlag=FALSE)
        if (method=='mice-pmm') {
          options$donors <- as.integer(param('donors',5))
          options$ridge <- param('ridge',1e-5)
        }
        do.call(mice::mice,options)
      }, enabled=param('baseline_engine','optimized')=='optimized')
      fit <- fitted$value
      acceleration <- fitted$acceleration
      execution$mice_version <- as.character(utils::packageVersion('mice'))
      execution$mice_imputations <- m
      execution$mice_iterations <- fit$iteration
      draws <- lapply(seq_len(m),function(i) as.matrix(mice::complete(fit,action=i)))
      avg <- Reduce('+',draws)/m
      if (method=='mice-pmm') avg else t(avg)
    },
    'missforest' = {
      need('missForest')
      options <- list(xmis=input,maxiter=param('max_iter',10),ntree=param('trees',100),
                      mtry=max(1L,floor(ncol(input)^(1/3))),verbose=FALSE)
      forest_threads <- as.integer(param('threads',1))
      # Small ranger tasks spend disproportionate time starting threads. Each
      # feature still gets every requested tree and the same native fit order.
      if (param('baseline_engine','optimized')=='optimized' && length(input) < 10000) {
        forest_threads <- 1L
        Sys.setenv(R_RANGER_NUM_THREADS='1')
        acceleration <- list(applied=TRUE, reason='serial small ranger tasks; native tree count and fit order')
      }
      if ('num.threads' %in% names(formals(missForest::missForest))) options$num.threads <- forest_threads
      execution$forest_threads <- forest_threads
      execution$trees <- options$ntree
      execution$forest_max_iterations <- options$maxiter
      execution$missforest_version <- as.character(utils::packageVersion('missForest'))
      do.call(missForest::missForest,options)$ximp
    },
    'grr' = {
      if (requireNamespace('DreamAI',quietly=TRUE)) {
        f <- DreamAI::impute.RegImpute
      } else {
        need('glmnet')
        root <- if (length(args)>=7) args[7] else ''
        choices <- c(root,file.path(root,'Code','R','RegImpute.R'),file.path(root,'R','RegImpute.R'))
        src <- choices[file.exists(choices) & !dir.exists(choices)]
        if (!length(src)) stop('GRR requires DreamAI or its original RegImpute.R source')
        scope <- new.env(parent=globalenv()); sys.source(src[1],envir=scope)
        f <- scope$impute.RegImpute
      }
      t(f(data=x,fillmethod='row_mean',maxiter_RegImpute=param('max_iter',10),conv_nrmse=1e-3))
    },
    'gms' = {
      need('GMSimpute'); t(GMSimpute::GMS.Lasso(as.data.frame(x),nfolds=3,log.scale=FALSE,TS.Lasso=TRUE))
    },
    'trknn' = {
      choices <- c(file.path(args[6],'inst','NAguideRapp','Trunc_KNN','Imput_funcs.r'),
                   file.path(args[6],'Trunc_KNN','Imput_funcs.r'))
      src <- choices[file.exists(choices)]
      if (!length(src)) stop('trknn requires original Trunc_KNN/Imput_funcs.r via --naguider-source')
      scope <- new.env(parent=globalenv()); sys.source(src[1],envir=scope)
      t(scope$imputeKNN(x,k=min(k,nrow(x)-1L),distance='truncation',perc=0))
    },
    stop(paste('unsupported R method:',method)))
  result <- as.matrix(result); storage.mode(result) <- 'double'
  if (!identical(dim(input),dim(result))) stop('R method returned a mismatched shape')
  if (any(!is.finite(result))) stop('R method returned nonfinite predictions')
  result[is.finite(input)] <- input[is.finite(input)]
  jsonlite::write_json(list(acceleration=acceleration,execution=execution),paste0(args[3],'.json'),auto_unbox=TRUE)
  if (binary) {
    connection <- file(args[3], 'wb')
    tryCatch(writeBin(as.double(t(result)),connection,size=8,endian='little'),finally=close(connection))
  } else {
    utils::write.csv(result,args[3],row.names=FALSE,na='NA')
  }
},error=function(e) {
  cat(conditionMessage(e),'\n',file=stderr()); quit(status=1)
})
