# Multiclass learning the atmospheric labels

library(tidyverse)
library(caret)

library(doMC)
registerDoMC(cores = 4)

atmos_labels <- c("clear", "haze", "partly_cloudy", "cloudy")

X <- read_csv("X-summ-tif.csv",
              col_types = paste(rep("d", 24), collapse = ""))
Y <- read_csv("labelmat.csv") %>% select(one_of(atmos_labels)) %>% as.matrix()
Y_vec <- apply(Y, 1, function(a) colnames(Y)[a == 1])

D <- cbind(X, atmos = factor(Y_vec))

set.seed(1000)
train_ind <- sample(1:nrow(D), 36000)
D_train <- D[train_ind, ]
D_valid <- D[-train_ind, ]

## LR

library(glmnet)

glmnet_cntrl <- trainControl(method = "cv",
                             number = 5,
                             classProbs = TRUE,
                             allowParallel = TRUE)

glmnet_grid <- expand.grid(alpha = c(0, 1, 2), lambda = c(0, 1, 2))

glmnet_fit <- train(atmos ~ ., data = D_train,
                    method = "glmnet",
                    trControl = glmnet_cntrl,
                    tuneGrid = glmnet_grid, 
                    metric = "Kappa")

glmnet_pred <- predict(glmnet_fit, D_valid)
confusionMatrix(glmnet_pred, D_valid$atmos)

#### acc = 0.8759

## RF

library(randomForest)

rf_cntrl <- trainControl(method = "cv",
                         number = 5,
                         classProbs = TRUE,
                         allowParallel = TRUE)

rf_grid <- expand.grid(mtry = 3:7)

rf_fit <- train(atmos ~ ., data = D_train,
                method = "rf",
                    trControl = rf_cntrl,
                    tuneGrid = rf_grid)

rf_pred <- predict(rf_fit, D_valid)
confusionMatrix(rf_pred, D_valid$atmos)
#### acc = .8918

## trying MLR package
library(mlr)
library(parallelMap)
parallelStartMulticore(3)

atmos_task <- makeClassifTask(data = D, target = "atmos")
rf_lrn <- makeLearner("classif.randomForest", predict.type = "response", mtry = 5, ntree = 700) #predict.threshold, par.vals

train_ind <- sample(nrow(D), 36000)
valid_ind <- setdiff(1:nrow(D), train_ind)

params <- makeParamSet(
  makeDiscreteParam("mtry", 3:8),
  makeDiscreteParam("ntree", 100:1000)
)

ctrl <- makeTuneControlRandom(maxit = 100)

rdesc <- makeResampleDesc("CV", iters = 5, stratify = TRUE)

res <- tuneParams("classif.randomForest", task = atmos_task, resampling = rdesc,
                  par.set = params, control = ctrl)

r <- resample(rf_lrn, task, rdesc)
rf_fit <- train(rf_lrn, task, subset = train_ind)
pred <- predict(rf_fit, task, subset = valid_ind)
performance(pred, measures = list(mmce, acc))

parallelStop()

## XGB

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
                   nrounds = 1000,
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


bst_model <- xgb.train(params = xgb_params,
                       data = train_matrix,
                       nrounds = nround)
# Predict hold-out test set
test_pred <- predict(bst_model, newdata = test_matrix)
test_prediction <- matrix(test_pred, nrow = numberOfClasses,
                          ncol=length(test_pred)/numberOfClasses) %>%
  t() %>%
  data.frame() %>%
  mutate(label = test_label + 1,
         max_prob = max.col(., "last"))
# confusion matrix of test set
confusionMatrix(factor(test_prediction$label),
                factor(test_prediction$max_prob),
                mode = "everything")