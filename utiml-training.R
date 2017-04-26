# Utiml models
library(tidyverse)
library(utiml)
library(xgboost)
library(randomForest)

nCores <- detectCores()

## Fbeta

utiml_measure_f2 <- function (mlconfmat, ...) {
  prec <- mlconfmat$TPi/(mlconfmat$TPi + mlconfmat$FPi)
  rec <- mlconfmat$TPi/(mlconfmat$TPi + mlconfmat$FNi)
  sum((1+2^2) * (prec*rec)/(2^2 * prec + rec), na.rm = TRUE)/nrow(mlconfmat$Y)
}

## Preproc

X <- read_csv("/home/jan/data/X-sum-jpg.csv",
              col_types = paste(rep("d", 18), collapse = ""))
Y <- read_csv("/home/jan/data/labelmat.csv")

mldr_train <- mldr_from_dataframe(cbind(X, Y), labelIndices = (ncol(X)+1):(ncol(X)+ncol(Y)))

#mldr_train <- normalize_mldata(mldr_train)
#mldr_train <- remove_skewness_labels(mldr_train, 1000)

mdata <- create_holdout_partition(mldr_train, c(train = 0.9, valid = 0.1), "stratified")

## Training

br_rf_model <- br(mdata$train, "RF", ntrees = 100, cores = nCores)

## Validation

br_rf_prob_valid <- predict(br_rf_model, mdata$valid, cores = nCores)

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
