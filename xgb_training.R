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

# primary

param <- list("objective" = "binary:logistic",
              "scale_pos_weight" = 1,
              "eta" = 0.1,
              "gamma" = 0,
              "max_depth" = 5,
              "silent" = 1,
              "nthread" = 12,
              "lambda" = 1,
              "alpha" = 0,
              "subsample" = 1,
              "colsample_bytree" = 1)

xgmat_valid <- xgb.DMatrix(x_valid)
predictions <- matrix(0, ncol = ncol(y_valid), nrow = nrow(y_valid))
colnames(predictions) <- colnames(y_valid)

for(label in colnames(y_train)) {
  print(label)
  xgmat_train <- xgb.DMatrix(x_train, label = y_train[, label])
  #xgmat_valid <- xgb.DMatrix(x_valid, label = y_valid[, label])
  
  mod <- xgb.train(param, xgmat_train, nrounds = 200)
  predictions[, label] <- predict(mod, xgmat_valid)
}

newpred <- ifelse(predictions > 0.1, 1, 0)
#newpred <- pcut_threshold(predictions, ratio = apply(y_train, 2, mean)) %>% as.matrix()

F_score(y_valid, newpred)
