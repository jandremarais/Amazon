# Multiclass learning the atmospheric labels

library(tidyverse)
library(caret)

library(doMC)
registerDoMC(cores = 4)

atmos_labels <- c("clear", "haze", "partly_cloudy", "cloudy")

train_labels <- read_csv("data/train_v2.csv")

X <- read_csv("data/X-summ-tif.csv",
              col_types = paste(rep("d", 36), collapse = "")) %>% 
  filter(train_labels$image_name != "train_24448")# %>% 
  #as.matrix()
Y <- read_csv("data/labelmat.csv") %>% select(one_of(atmos_labels)) %>% 
  filter(train_labels$image_name != "train_24448") %>% 
  as.matrix()

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
parallelStartMulticore(12)

atmos_task <- makeClassifTask(data = D, target = "atmos")
rf_lrn <- makeLearner("classif.randomForest", predict.type = "response", mtry = 5, ntree = 700) #predict.threshold, par.vals

train_ind <- sample(nrow(D), 36000)
valid_ind <- setdiff(1:nrow(D), train_ind)

params <- makeParamSet(
  makeDiscreteParam("mtry", 3:8),
  makeDiscreteParam("ntree", 100:1000)
)

ctrl <- makeTuneControlRandom(maxit = 50)

rdesc <- makeResampleDesc("CV", iters = 3, stratify = TRUE)

res <- tuneParams("classif.randomForest", task = atmos_task, resampling = rdesc,
                  par.set = params, control = ctrl)

r <- resample(rf_lrn, task, rdesc)
rf_fit <- train(rf_lrn, atmos_task, subset = train_ind)
pred <- predict(rf_fit, atmos_task, subset = valid_ind)
performance(pred, measures = list(mmce, acc))

parallelStop()

## XGB

y <- factor(Y_vec, levels = c("clear", "haze", "partly_cloudy", "cloudy"))
y <- as.numeric(y) - 1 

library(xgboost)
library(parallel)
nCores <- detectCores()

oversamp <- function(data, response) {
  y <- data[, response]
  counts <- table(y)
  max_lab <- names(counts)[counts == max(counts)]
  rbind(data[y == max_lab, ],
        do.call("rbind", lapply(names(counts[-1]), function(a) {
          data[y == a, ][sample(counts[a], max(counts), replace = TRUE), ]
        })))
}

D_train_over <- oversamp(D_train, response = "atmos")

train_matrix <- xgb.DMatrix(as.matrix(D_train_over[, -ncol(D_train_over)]), label = as.numeric(D_train_over$atmos) - 1)
valid_matrix <- xgb.DMatrix(as.matrix(D_valid[, -ncol(D_valid)]), label = as.numeric(D_valid$atmos) - 1)

nClasses <- ncol(Y)
xgb_params <- list("objective" = "multi:softprob",
                   #"eval_metric" = "mlogloss",
                   "num_class" = nClasses,
                   "eta" = 0.1,
                   "max_depth" = 15,
                   "subsample" = 0.8,
                   "colsample_by_tree" = 0.8,
                   "lambda" = 1)

bst_model <- xgb.train(params = xgb_params,
                       data = train_matrix,
                       nrounds = 1000,
                       print_every_n = 10,
                       verbose = TRUE, 
                       watchlist = list(train = train_matrix, valid = valid_matrix))

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