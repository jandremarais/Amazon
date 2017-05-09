# data set creation

# to add to ipython notebook
# y_train = np.array(y_train, np.uint8)
# x_train = np.array(x_train, np.float16) / 255.
# x_train = x_train[:,0,:]
# np.savetxt("vgg16_probs_train.csv", x_train, delimiter=",")

path <- "data"

## Label matrix

library(tidyverse)
library(stringr)

train_labels <- read_csv("data/train_v2.csv")
label_list <- strsplit(train_labels$tags, " ")
L <- unique(unlist(label_list))

Y <- ifelse(t(sapply(label_list, function(a) L %in% a)), 1, 0)
colnames(Y) <- L
#Y <- data.frame(id = train_labels$image_name, Y, stringsAsFactors = FALSE)

# train_labels[apply(Y[,c("clear", "cloudy", "partly_cloudy", "haze")], 1, sum) == 0,]
# note image train_24448 has no atmospheric labels -> should probably remove

write_csv(data.frame(Y), "data/labelmat.csv")

## Features

### JPG

#### Summary statistics

library(raster)
library(pbapply)
library(moments)
library(parallel)

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

temp <- c(imageData(readImage("data/train-tif-v2/train_0.tif"))[,,1])
c(#mean = mean(dat),
  summary(temp),
  sd = sd(temp),
  #min = min(dat),
  #max = max(dat),
  kurt = kurtosis(temp),
  skew = skewness(temp))

X_list <- mclapply(paste0("data/train-tif-v2/", train_labels$image_name, ".tif"), function(a) {
  img <- readImage(a)
  img_data <- imageData(img)
  c(sapply(1:4, function(b) {
    dat <- c(img_data[,,b])
    c(#mean = mean(dat),
      summary(dat),
      sd = sd(dat),
      #min = min(dat),
      #max = max(dat),
      kurt = kurtosis(dat),
      skew = skewness(dat))
  }))
}, mc.cores = nCores)

X <- do.call("rbind", X_list)

#colnames(X) <- c(sapply(c("R", "G", "B", "NIR"), function(a) paste(a, c("mean", "sd", "min", "max", "kurtosis", "skewness"), sep = "_")))
colnames(X) <- c(sapply(c("R", "G", "B", "NIR"), 
                        function(a) paste(a, c("min", "Q1", "median", "mean", "Q3", "max", "sd", "kurtosis", "skewness"), sep = "_")))
write_csv(data.frame(X), "data/X-summ-tif.csv")

## test
X_list <- mclapply(paste(path, "test-tif-v2", list.files(paste0(path, "/test-tif-v2/")), sep = "/"), function(a) {
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

write_csv(data.frame(X), paste(path, "X-summ-tif-test.csv", sep = "/"))

