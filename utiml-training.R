# Utiml models
library(tidyverse)
library(utiml)
library(xgboost)
library(randomForest)
library(e1071)
library(kknn)
library(parallel)
library(stringr)

nCores <- detectCores()

## Fbeta

utiml_measure_f2 <- function (mlconfmat, ...) {
  prec <- mlconfmat$TPi/(mlconfmat$TPi + mlconfmat$FPi)
  rec <- mlconfmat$TPi/(mlconfmat$TPi + mlconfmat$FNi)
  sum((1+2^2) * (prec*rec)/(2^2 * prec + rec), na.rm = TRUE)/nrow(mlconfmat$Y)
}

Fb_score <- function(act_mat, pred_mat, B = 2) {
  mean(sapply(1:nrow(act_mat), function(a) {
    tp <- sum(pred_mat[a, ] & act_mat[a, ])
    
    prec <- tp/sum(pred_mat[a, ])
    rec <- tp/sum(act_mat[a, ])
    
    ifelse(tp != 0, (1+B^2) * (prec*rec)/(B^2 * prec + rec), 0)
  }))
}

## Preproc

X <- read_csv("X-summ-jpg.csv",
              col_types = paste(rep("d", 18), collapse = ""))

X <- read_csv("data/X-summ-tif.csv",
              col_types = paste(rep("d", 24), collapse = ""))

X <- read_csv("/home/jan/data/X-sum-jpg.csv",
              col_types = paste(rep("d", 18), collapse = ""))

Y <- read_csv("data/labelmat.csv")

mldr_train <- mldr_from_dataframe(cbind(X, Y), labelIndices = (ncol(X)+1):(ncol(X)+ncol(Y)))

#mldr_train <- normalize_mldata(mldr_train)
#mldr_train <- remove_skewness_labels(mldr_train, 1000)

set.seed(1234)
mdata <- create_holdout_partition(mldr_train, c(train = 0.9, valid = 0.1), "stratified")

## Training

br_rf_model <- br(mdata$train, "RF", ntrees = 700, mtry = 5, cores = nCores)

br_knn_model <- br(mdata$train, "KNN", k = 30, cores = nCores)

br_xgb_model <- br(mdata$train, "XGB", 
                   nrounds = 1000,
                   eta = 0.01,
                   objective = "binary:logistic",
                   max_depth = 10,
                   cores = nCores)

br_svm_model <- br(mdata$train, "SVM", kernel = "radial", cores = nCores)

mbr_rf_model <- mbr(mdata$train, "RF", ntrees = 100, cores = nCores)

brp_rf_model <- brplus(mdata$train, "RF", ntrees = 200, cores = nCores) # very poor

cc_rf_model <- cc(mdata$train, "RF", ntrees = 100, cores = nCores, 
                  chain = names(sort(apply(Y, 2, sum), decreasing = TRUE)))

## Validation

### Func to determine optimal threshold

get_opt_cut <- function(probs, valid, try_these) {
  unlist(mclapply(1:ncol(probs), function(a) {
    temp <- lapply(try_these, function(b) {
      thresh <- rep(0.5, ncol(probs))
      thresh[a] <- b
      preds <- fixed_threshold(probs, thresh)
      #confmat <- multilabel_confusion_matrix(valid, preds)
      #utiml_measure_f2(confmat)
      Fb_score(as.matrix(valid), as.matrix(preds))
    })
    temp <- unlist(temp)
    try_these[temp == max(temp)][1]
  }, mc.cores = detectCores()))
}

Fb_score(mdata$valid$dataset[, mdata$valid$labels$index])

###

br_rf_prob_valid <- predict(br_rf_model, mdata$valid, cores = nCores)
br_rf_threshs <- get_opt_cut(br_rf_prob_valid, mdata$valid$dataset[, mdata$valid$labels$index], seq(0.02, 0.4, 0.02))
br_rf_pred_valid <- fixed_threshold(br_rf_prob_valid, br_rf_threshs)

br_xgb_prob_valid <- predict(br_xgb_model, mdata$valid, cores = nCores)
br_xgb_threshs <- get_opt_cut(br_xgb_prob_valid, mdata$valid, seq(0.02, 0.4, 0.02))
br_xgb_pred_valid <- fixed_threshold(br_xgb_prob_valid, br_xgb_threshs)

br_knn_prob_valid <- predict(br_knn_model, mdata$valid, cores = nCores)
br_knn_threshs <- get_opt_cut(br_knn_prob_valid, mdata$valid, seq(0.02, 0.4, 0.02))
br_knn_pred_valid <- fixed_threshold(br_knn_prob_valid, br_knn_threshs)

br_svm_prob_valid <- predict(br_svm_model, mdata$valid, cores = nCores)
br_svm_threshs <- get_opt_cut(br_svm_prob_valid, mdata$valid, seq(0.02, 0.4, 0.02))
br_svm_pred_valid <- fixed_threshold(br_svm_prob_valid, br_svm_threshs)

mbr_rf_prob_valid <- predict(mbr_rf_model, mdata$valid, cores = nCores)
mbr_rf_pred_valid <- fixed_threshold(mbr_rf_prob_valid, 0.02)

brp_rf_prob_valid <- predict(brp_rf_model, mdata$valid, cores = nCores)
brp_rf_pred_valid <- fixed_threshold(brp_rf_prob_valid, 0.01)

cc_rf_prob_valid <- predict(cc_rf_model, mdata$valid, cores = nCores)
cc_rf_pred_valid <- fixed_threshold(cc_rf_prob_valid, 0.25)

ensemble_prob_valid <- (br_xgb_prob_valid+br_knn_prob_valid+br_rf_prob_valid)/3
threshs <- get_opt_cut(ensemble_prob_valid, mdata$valid, seq(0.02, 0.3, 0.02))
ensemble_pred_valid <- fixed_threshold(ensemble_prob_valid, threshs)

ensemble_pred_valid <- ifelse((br_xgb_pred_valid + 
                                 br_rf_pred_valid + 
                                 br_knn_pred_valid + 
                                 br_svm_pred_valid) >= 2, 1, 0)

confmat <- multilabel_confusion_matrix(mdata$valid, ensemble_pred_valid)
utiml_measure_f2(confmat)

Fb_score(as.matrix(mdata$valid$dataset[, mdata$valid$labels$index]), as.matrix(br_rf_pred_valid))

Fb_score(as.matrix(mdata$valid$dataset[, mdata$valid$labels$index])[, c("cloudy", "haze", "partly_cloudy", "clear")], 
         as.matrix(br_rf_pred_valid)[, c("cloudy", "haze", "partly_cloudy", "clear")])

confmat <- multilabel_confusion_matrix(mdata$valid, br_rf_pred_valid)
utiml_measure_f2(confmat)

# best is 0.8878513 for br(rf,svm,xgb,knn) ensemble with X-sum

## Prediction

X_test <- read_csv("/home/jan/data/X-sum-jpg-test.csv", 
                   col_types = paste(rep("d", 18), collapse = ""))

br_rf_prob_test <- predict(br_rf_model, X_test, cores = nCores)
br_rf_pred_test <- ifelse(br_rf_prob_test > 0.2, 1, 0)

br_rf_pred_sub <- sapply(1:nrow(X_test), function(a) {
  paste(colnames(Y)[br_rf_prob_test[a, ] == 1], collapse = " ")
})

br_rf_subm <- data.frame(image_name = str_replace_all(list.files("/home/jan/data/test-jpg"), ".jpg", ""),
                   tags = br_rf_pred_sub)

write_csv(br_rf_subm, "/home/jan/data/subm1.csv")

###############################

br_xgb_prob_test <- predict(br_xgb_model, X_test, cores = nCores)
br_xgb_pred_test <- as.matrix(fixed_threshold(br_xgb_prob_test, threshs))

br_xgb_pred_sub <- sapply(1:nrow(X_test), function(a) {
  paste(colnames(Y)[br_xgb_pred_test[a, ] == 1], collapse = " ")
})

br_xgb_subm <- data.frame(image_name = str_replace_all(list.files("/home/jan/data/test-jpg"), ".jpg", ""),
                         tags = br_xgb_pred_sub)

write_csv(br_xgb_subm, "/home/jan/data/subm-br-xgb.csv")
