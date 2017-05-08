# data set creation

path <- "."

## Label matrix

library(tidyverse)
library(stringr)

train_labels <- read_csv(paste(path, "train.csv", sep = "/"))
label_list <- strsplit(train_labels$tags, " ")
L <- unique(unlist(label_list))

Y <- ifelse(t(sapply(label_list, function(a) L %in% a)), 1, 0)
colnames(Y) <- L
#Y <- data.frame(id = train_labels$image_name, Y, stringsAsFactors = FALSE)

write_csv(data.frame(Y), "labelmat.csv")

## Features

### JPG

#### Summary statistics

library(raster)
library(pbapply)
library(moments)
library(parallel)

paste0(path, "/train-jpg/", train_labels$image_name, ".jpg")

#source("https://bioconductor.org/biocLite.R")
#biocLite("EBImage")

library(EBImage)

nCores <- detectCores()

X_list <- mclapply(paste0(path, "/train-jpg/", train_labels$image_name, ".jpg"), function(a) {
  img <- readImage(a)
  img_data <- imageData(img)
  c(sapply(1:3, function(b) {
    dat <- c(img_data[,,b])
    c(mean = mean(dat),
      sd = sd(dat),
      min = min(dat),
      max = max(dat),
      kurt = kurtosis(dat),
      skew = skewness(dat))
  }))
}, mc.cores = nCores)

X <- do.call("rbind", X_list)

colnames(X) <- c(sapply(c("R", "G", "B"), function(a) paste(a, c("mean", "sd", "min", "max", "kurtosis", "skewness"), sep = "_")))
write_csv(data.frame(X), "X-summ-jpg.csv")

### TIF

#### Summary statistics

X_list <- mclapply(paste0(path, "/train-tif-v2/", train_labels$image_name, ".tif"), function(a) {
  img <- readImage(a)
  img_data <- imageData(img)
  c(sapply(1:4, function(b) {
    dat <- c(img_data[,,b])
    c(mean = mean(dat),
      sd = sd(dat),
      min = min(dat),
      max = max(dat),
      kurt = kurtosis(dat),
      skew = skewness(dat))
  }))
}, mc.cores = nCores)

X <- do.call("rbind", X_list)

colnames(X) <- c(sapply(c("R", "G", "B", "NIR"), function(a) paste(a, c("mean", "sd", "min", "max", "kurtosis", "skewness"), sep = "_")))
write_csv(data.frame(X), "X-summ-tif.csv")

img_temp <- readImage("./train-tif-v2/train_1.tif")
plot(img_temp)
