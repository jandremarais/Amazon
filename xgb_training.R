# training xgboost model for each label

library(tidyverse)
library(stringr)

X <- read_csv("/home/jan/data/X-sum-jpg.csv",
              col_types = paste(rep("d", 18), collapse = "")) %>% as.matrix()

Y <- read_csv("/home/jan/data/labelmat.csv") %>% as.matrix()

train_ind <- sample(1:nrow(X), 36000)
x_train <- X[train_ind, ]
x_valid <- X[-train_ind, ]

y_train <- Y[train_ind, ]
y_valid <- Y[-train_ind, ]

library(xgboost)
library(parallel)
nCores <- detectCores()

library(utiml)

# primary

label = "primary"
xgmat_train <- xgb.DMatrix(x_train, label = y_train[, label])
xgmat_valid <- xgb.DMatrix(x_valid, label = y_valid[, label])

param_primary <- list("objective" = "binary:logitraw",
              "eval_metric" = "auc",
              "eval_metric" = "error",
              "scale_pos_weight" = 1,
              "eta" = 0.01,
              "gamma" = 0,
              "max_depth" = 10,
              "silent" = 1,
              "nthread" = -1,
              "lambda" = 1,
              "alpha" = 0,
              "subsample" = 1,
              "colsample_bytree" = 0.8,
              "min_child_weight" = 2)

mod_primary <- xgb.train(param_primary, xgmat_train, nrounds = 700, print_every_n = 10, 
                 watchlist = list(train = xgmat_train, valid = xgmat_valid))

F_score(y_valid, newpred)

library(caret)

xgbGrid <- expand.grid(
  nrounds = seq(400, 800, 100),
  max_depth = seq(6, 16, 2),
  eta = 0.01,
  gamma = c(0, 1, 2),
  colsample_bytree = 0.8,
  min_child_weight = c(1, 2),
  subsample = 1
)

xgbTrControl <- trainControl(
  method = "cv",
  number = 5,
  #repeats = 2,
  verboseIter = FALSE,
  returnData = FALSE,
  allowParallel = TRUE
)

library(doMC)
registerDoMC(cores = 12)

temp <- lapply(colnames(Y), function(a) {
  xgbTrain <- train(
    x = x_train, 
    y = factor(y_train[, a]),
    objective = "binary:logistic",
    trControl = xgbTrControl,
    tuneGrid = xgbGrid,
    method = "xgbTree"
  )
})

best_params <- lapply(temp, "[[", "bestTune")
best_params <- lapply(best_params, as.list)
best_params <- lapply(best_params, function(a) {
  c(a[-1],
    "objective" = "binary:logistic",
    "nthread" = -1,
    "silent" = 1,
    "eval_metric" = "auc")
})

best_nrounds <- sapply(temp, function(a) a$bestTune$nrounds)

best_xgb <- pblapply(1:ncol(Y), function(a) {
  label = colnames(Y)[a]
  xgmat_train <- xgb.DMatrix(x_train, label = y_train[, label])
  xgmat_valid <- xgb.DMatrix(x_valid, label = y_valid[, label])
  
  xgb.train(best_params[[a]], xgmat_train, nrounds = best_nrounds[a]) 
            #watchlist = list(train = xgmat_train, valid = xgmat_valid),
            #early_stopping_rounds = 50, print_every_n = 10)
})

probs <- sapply(1:ncol(Y), function(a) {
  label = colnames(Y)[a]
  xgmat_valid <- xgb.DMatrix(x_valid, label = y_valid[, label])
  predict(best_xgb[[a]], xgmat_valid)
})

library(utiml)
mldr_valid <- mldr_from_dataframe(data.frame(x_valid, y_valid), labelIndices = (ncol(x_valid)+1):(ncol(x_valid)+ncol(y_valid)))
threshs <- get_opt_cut(probs, mldr_valid, seq(0.01, 0.4, 0.025))
preds <- fixed_threshold(probs, 0.22)
confmat <- multilabel_confusion_matrix(mldr_valid, preds)
utiml_measure_f2(confmat)
