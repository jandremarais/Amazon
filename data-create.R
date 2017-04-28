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

library(raster)
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


## Resize image to 32x32

path <- '/home/jan/data/train-jpg'
img <- brick(paste(path, "train_0.jpg", sep = "/"))
svd_out <- svd(getValues(img$train_0.1, format = "matrix"))
svd_out$u[, 1:64] %*% diag(svd_out$d[1:64]) %*% t(svd_out$v[, 1:64])

ind <- seq(0, 256, 8)
lapply(2:(length(ind)), function(a) {
  width <- (ind[a-1]+1):(ind[a])
  mean(values(img$train_0.1, format = "matrix")[width, width])
})

X_list <- mclapply(list.files(path)[1:10], function(a) {
  img <- brick(paste(path, a, sep = "/"))
  c(values(img))
}, mc.cores = detectCores())

X <- do.call("rbind", X_list)
write_csv(data.frame(X), "/home/jan/data/X-sum-jpg.csv")


## creaing a dataset from the histograms of the images using EBImage

source("https://bioconductor.org/biocLite.R")
biocLite("EBImage")

library(EBImage)
library(pbapply)
library(parallel)

nCores <- detectCores()
path <- '/home/jan/data/train-jpg'

X_list <- mclapply(list.files(path), function(a) {
  img <- readImage(paste(path, a, sep = "/"))
  hist_data <- hist(img)
  unlist(lapply(hist_data, "[[", "density"))
}, mc.cores = nCores)

X <- do.call("rbind", X_list)
write_csv(data.frame(X), "/home/jan/data/X-hist-jpg.csv")

path <- '/home/jan/data/test-jpg'

X_list <- mclapply(list.files(path), function(a) {
  img <- readImage(paste(path, a, sep = "/"))
  hist_data <- hist(img)
  unlist(lapply(hist_data, "[[", "density"))
}, mc.cores = detectCores())

X_test <- do.call("rbind", X_list)
write_csv(data.frame(X_test), "/home/jan/data/X-hist-jpg-test.csv")
