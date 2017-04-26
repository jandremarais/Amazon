# data set creation

## Label matrix

library(tidyverse)
library(stringr)

train_labels <- read_csv("/home/jan/data/train.csv")
rearrange_ind <- match(str_replace_all(list.files("/home/jan/data/train-jpg"), ".jpg", ""),
                       train_labels$image_name)
train_labels <- train_labels[rearrange_ind, ]
label_list <- strsplit(train_labels$tags, " ")
L <- unique(unlist(label_list))

Y <- sapply(L, function(b) sapply(label_list, function(a) ifelse(b %in% a, 1, 0)))
write_csv(data.frame(Y), "/home/jan/data/labelmat.csv")

## Basic features

library(pbapply)
library(moments)
library(parallel)

path <- '/home/jan/data/train-jpg'

X_list <- mclapply(list.files(path), function(a) {
  img <- brick(paste(path, a, sep = "/"))
  c(apply(values(img), 2, function(a) {
    c(mean = mean(a),
      sd = sd(a),
      min = min(a),
      max = max(a),
      kurt = kurtosis(a),
      skew = skewness(a))
  }))
}, mc.cores = detectCores())

X <- do.call("rbind", X_list)
write_csv(data.frame(X), "/home/jan/data/X-sum-jpg.csv")

path <- '/home/jan/data/test-jpg'

X_list <- mclapply(list.files(path), function(a) {
  img <- brick(paste(path, a, sep = "/"))
  c(apply(values(img), 2, function(a) {
    c(mean = mean(a),
      sd = sd(a),
      min = min(a),
      max = max(a),
      kurt = kurtosis(a),
      skew = skewness(a))
  }))
}, mc.cores = detectCores())

X_test <- do.call("rbind", X_list)
write_csv(data.frame(X_test), "/home/jan/data/X-sum-jpg-test.csv")