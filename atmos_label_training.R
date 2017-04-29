library(tidyverse)
library(stringr)

X <- read_csv("/home/jan/data/X-sum-jpg.csv",
              col_types = paste(rep("d", 18), collapse = "")) %>% as.matrix()

Y <- read_csv("/home/jan/data/labelmat.csv") %>% 
  select(haze, clear, partly_cloudy, cloudy) %>% 
  as.matrix()

y <- apply(Y, 1, function(a) colnames(Y)[a == 1])
y <- factor(y, levels = c("clear", "haze", "partly_cloudy", "cloudy"))
y <- as.numeric(y) - 1 

train_ind <- sample(1:nrow(X), 36000)
x_train <- X[train_ind, ]
x_valid <- X[-train_ind, ]

y_train <- y[train_ind]
y_valid <- y[-train_ind]

library(xgboost)
library(parallel)
nCores <- detectCores()

train_matrix <- xgb.DMatrix(x_train, label = y_train)
valid_matrix <- xgb.DMatrix(x_valid, label = y_valid)

nClasses <- max(y) + 1
xgb_params <- list("objective" = "multi:softprob",
                   "eval_metric" = "mlogloss",
                   "num_class" = nClasses,
                   "eta" = 0.1,
                   "max_depth" = 10)

cv_model <- xgb.cv(params = xgb_params,
                   data = train_matrix, 
                   nrounds = 50,
                   nfold = 5,
                   verbose = FALSE,
                   prediction = TRUE)

OOF_prediction <- data.frame(cv_model$pred) %>%
  mutate(max_prob = max.col(., ties.method = "last"),
         label = y_train + 1)

library(caret)
confusionMatrix(factor(OOF_prediction$label), 
                factor(OOF_prediction$max_prob),
                mode = "everything")
