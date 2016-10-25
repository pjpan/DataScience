h2o.h2oEnsemble <- function (x, y, training_frame, model_id = NULL, validation_frame = NULL, 
    family = c("AUTO", "binomial", "gaussian"), learner = c("h2o.glm.wrapper", 
        "h2o.randomForest.wrapper", "h2o.gbm.wrapper", "h2o.deeplearning.wrapper"), 
    metalearner = "h2o.glm.wrapper", cvControl = list(V = 5, 
        shuffle = TRUE), seed = 1, parallel = "seq", keep_levelone_data = TRUE) 
{
    starttime <- Sys.time()
    runtime <- list()
    if ((!inherits(training_frame, "Frame") && !inherits(training_frame, 
        "H2OFrame"))) 
        tryCatch(training_frame <- h2o.getFrame(training_frame), 
            error = function(err) {
                stop("argument \"training_frame\" must be a valid H2OFrame or id")
            })
    if (!is.null(validation_frame)) {
        if (is.character(validation_frame)) 
            tryCatch(validation_frame <- h2o.getFrame(validation_frame), 
                error = function(err) {
                  stop("argument \"validation_frame\" must be a valid H2OFrame or id")
                })
    }
    N <- dim(training_frame)[1L]
    if (is.null(validation_frame)) {
        validation_frame <- training_frame
    }
    if (length(family) > 0) {
        family <- match.arg(family)
    }
    if (family == "AUTO") {
        if (is.factor(training_frame[, y])) {
            numcats <- length(h2o.levels(training_frame[, y]))
            if (numcats == 2) {
                family <- "binomial"
            }
            else {
                stop("Multinomial case not yet implemented for h2o.ensemble. Check here for progress: https://0xdata.atlassian.net/browse/PUBDEV-2355")
            }
        }
        else {
            family <- "gaussian"
        }
    }
    if (family == c("gaussian")) {
        if (!is.numeric(training_frame[, y])) {
            stop("When `family` is gaussian, the repsonse column must be numeric.")
        }
        ylim <- c(min(training_frame[, y]), max(training_frame[, 
            y]))
    }
    else {
        if (!is.factor(training_frame[, y])) {
            stop("When `family` is binomial, the repsonse column must be a factor.")
        }
        else {
            numcats <- length(h2o.levels(training_frame[, y]))
            if (numcats > 2) {
                stop("Multinomial case not yet implemented for h2o.ensemble. Check here for progress: https://0xdata.atlassian.net/browse/PUBDEV-2355")
            }
        }
        ylim <- NULL
    }
    cvControl <- do.call(".cv_control", cvControl)
    V <- cvControl$V
    L <- length(learner)
    idxs <- expand.grid(1:V, 1:L)
    names(idxs) <- c("v", "l")
    if (length(metalearner) > 1 | !is.character(metalearner) | 
        !exists(metalearner)) {
        stop("The 'metalearner' argument must be a string, specifying the name of a base learner wrapper function.")
    }
    if (sum(!sapply(learner, exists)) > 0) {
        stop("'learner' function name(s) not found.")
    }
    if (!exists(metalearner)) {
        stop("'metalearner' function name not found.")
    }
    if (!(family %in% c("binomial", "gaussian"))) {
        stop("'family' not supported")
    }
    if (inherits(parallel, "character")) {
        if (!(parallel %in% c("seq", "multicore"))) {
            stop("'parallel' must be either 'seq' or 'multicore' or a snow cluster object")
        }
    }
    else if (!inherits(parallel, "cluster")) {
        stop("'parallel' must be either 'seq' or 'multicore' or a snow cluster object")
    }
    if (is.numeric(seed)) 
        set.seed(seed)
    folds <- sample(rep(seq(V), ceiling(N/V)))[1:N]
    training_frame$fold_id <- as.h2o(folds)
    if (grepl("^SL.", metalearner)) {
        metalearner_type <- "SuperLearner"
    }
    else if (grepl("^h2o.", metalearner)) {
        metalearner_type <- "h2o"
    }
    mm <- .make_Z(x = x, y = y, training_frame = training_frame, 
        family = family, learner = learner, parallel = parallel, 
        seed = seed, V = V, L = L, idxs = idxs, metalearner_type = metalearner_type)
    basefits <- mm$basefits
    Z <- mm$Z
    print("Metalearning")
    if (is.numeric(seed)) 
        set.seed(seed)
    if (grepl("^SL.", metalearner)) {
        if (is.character(family)) {
            familyFun <- get(family, mode = "function", envir = parent.frame())
        }
        Zdf <- as.data.frame(Z)
        Y <- as.data.frame(training_frame[, c(y)])[, 1]
        runtime$metalearning <- system.time(metafit <- match.fun(metalearner)(Y = Y, 
            X = Zdf, newX = Zdf, family = familyFun, id = seq(N), 
            obsWeights = rep(1, N)), gcFirst = FALSE)
    }
    else {
        Z$y <- training_frame[, c(y)]
        runtime$metalearning <- system.time(metafit <- match.fun(metalearner)(x = learner, 
            y = "y", training_frame = Z, validation_frame = NULL, 
            family = family), gcFirst = FALSE)
    }
    runtime$baselearning <- NULL
    runtime$total <- Sys.time() - starttime
    if (!keep_levelone_data) {
        Z <- NULL
    }
    out <- list(x = x, y = y, family = family, learner = learner, 
        metalearner = metalearner, cvControl = cvControl, folds = folds, 
        ylim = ylim, seed = seed, parallel = parallel, basefits = basefits, 
        metafit = metafit, levelone = Z, runtime = runtime, h2o_version = packageVersion(pkg = "h2o"), 
        h2oEnsemble_version = packageVersion(pkg = "h2oEnsemble"))
    class(out) <- "h2o.ensemble"
    return(out)
}