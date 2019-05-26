library(OpenImageR)
library(tidyverse)

library(e1071)
library(randomForest)

library(fastAdaboost)
library(gbm)


# Generate files paths
file_dir <- dirname(rstudioapi::getSourceEditorContext()$path)
train_path <- file.path(file_dir,"g1_train")
test_path <- file.path(file_dir,"g1_test")

# get training and testing file names
train_files <- list.files(train_path)
test_files <- list.files(test_path)

##########################################

# feature generation function
features <- function(file_path, file_name){
  path <- file.path(file_path, file_name)
  
  im <- readImage(path)
  #dim(im)
  
  im <- rgb_2gray(im)
  
  #imageShow(im)
  
  #intBl = resizeImage(im, width = 100, height = 100, method = 'bilinear')
  #dim(intBl)
  
  #im = im * 255
  
  hog <- HOG(im, cells = 4, orientations = 6)
  return(hog)
}

ans <- read_csv(file.path(file_dir,"train.csv"))

##########################################

len<-length(features(train_path,train_files[1]))
cn<-character()
for(a in 1:len){
  cn <- append(cn,paste("x",as.character(a), sep = ""))
}
cn <- append(cn,"y")

X <- matrix(NA, nrow=length(train_files), ncol=len+1)
i <- 1
for(file in train_files){
  X[i,] <- append(features(train_path,file),ans$has_cactus[ans$id==file])
  i <- i+1
}
colnames(X)=cn
#X <- as_tibble(X,colnames=cn)
#X <- X %>%
#  mutate(y = factor(y))

X <- as.data.frame(X)
X$y <- as.factor(X$y)

set.seed(Sys.time())
train_idx <- sample(1:nrow(X),11200)

oob_err<-double(11)
val_err<-double(11)

#mtry is no of Variables randomly chosen at each split
for(mtry in 5:15){
  rf<-randomForest(y ~ . , data = X , subset = train_idx,mtry=mtry,ntree=500) 
  oob_err[mtry-4] <- rf$err.rate[500] #Error of all Trees fitted
  
  pred<-predict(rf,X[-train_idx,])
  val_err[mtry-4]<- with(X[-train_idx,],mean(y!=pred))
}
# mtry was best at 14



val_err <- double(4)
i <- 1
for(k_type in c("linear", "sigmoid", "polynomial", "radial")){
  svmfit <- svm(y~.,data=X, kernel=k_type, subset=train_idx)
  pred<-predict(svmfit,X[-train_idx,])
  val_err[i]<- with(X[-train_idx,],mean(y!=pred))
  i = i+1
}
# output: l:0.10892857, s:0.20535714, p:0.13785714, r:0.07785714 => Radial was best

val_err <- double(5)
i <- 1
for(k in c(3,4,5,6,7)){
  svmfit <- svm(y~.,data=X, kernel="radial", cost=k, subset=train_idx)
  pred<-predict(svmfit,X[-train_idx,])
  val_err[i]<- with(X[-train_idx,],mean(y!=pred))
  i = i+1
}
# First run w/ c(.1,.5,1,5,10) => 5 was best at 0.07321429
# Second run w/ c(3,4,5,6,7) => 4,5,6 were all equal & the same so 5 what we will go with

rf <- randomForest(y~.,data=X, ntrees=1500, mtry=14, subset=train_idx)

rf_pred<-predict(rf,X[-train_idx,])
rf_pred_1<-predict(rf,filter(X[-train_idx,],y=="1"))
rf_pred_0<-predict(rf,filter(X[-train_idx,],y=="0"))
rf_err<-with(X[-train_idx,],mean(y!=rf_pred))
rf_err_1<- with(filter(X[-train_idx,],y=="1"),mean(y!=rf_pred_1))
rf_err_0<- with(filter(X[-train_idx,],y=="0"),mean(y!=rf_pred_0))

svm_model <- svm(y~.,data=X, kernel="radial", cost=5, subset=train_idx)

svm_pred<-predict(svm_model,X[-train_idx,])
svm_pred_1<-predict(svm_model,filter(X[-train_idx,],y=="1"))
svm_pred_0<-predict(svm_model,filter(X[-train_idx,],y=="0"))
svm_err<- with(X[-train_idx,],mean(y!=svm_pred))
svm_err_1<- with(filter(X[-train_idx,],y=="1"),mean(y!=svm_pred_1))
svm_err_0<- with(filter(X[-train_idx,],y=="0"),mean(y!=svm_pred_0))

######################################################
#                      Week 2                        #
######################################################

pc <- prcomp(select(X,-y))  # Eigen decomposition of Sample Covariance
pc1 <- prcomp(select(X,-y), scale. = T)  # Eigen decomposition of Sample Covariance

summary(pc)

Xpc = as_tibble(pc$x[, 1:55])
Xpc$y = X$y

## work around bug in gbm 2.1.1
predict.gbm <- function (object, newdata, n.trees, type = "link", single.tree = FALSE, ...) {
  if (missing(n.trees)) {
    if (object$train.fraction < 1) {
      n.trees <- gbm.perf(object, method = "test", plot.it = FALSE)
    }
    else if (!is.null(object$cv.error)) {
      n.trees <- gbm.perf(object, method = "cv", plot.it = FALSE)
    }
    else {
      n.trees <- length(object$train.error)
    }
    cat(paste("Using", n.trees, "trees...\n"))
    gbm::predict.gbm(object, newdata, n.trees, type, single.tree, ...)
  }
}

X2 <- as.data.frame(X)


X2$y <- as.character(X2$y)

adb <- gbm(y~.,data=X2[train_idx,], distribution = "bernoulli", n.trees = 500)
adb_pred<-predict(adb,X2[-train_idx,],n.trees=100, type = "response")
adb_pred_1<-predict(adb,filter(X2[-train_idx,],y=="1"),n.trees=500, type = "response")
adb_pred_0<-predict(adb,filter(X2[-train_idx,],y=="0"),n.trees=500, type = "response")

class_con <- function(x){
  if(x >= .5){
    return("1")
  }
  if(x < .5){
    return("0")
  }
}

adb_pred <- lapply(adb_pred, class_con)
adb_pred_1 <- lapply(adb_pred_1, class_con)
adb_pred_0 <- lapply(adb_pred_0, class_con)

adb_err<- with(X[-train_idx,],mean(y!=adb_pred))
adb_err_1<- with(filter(X[-train_idx,],y=="1"),mean(y!=adb_pred_1))
adb_err_0<- with(filter(X[-train_idx,],y=="0"),mean(y!=adb_pred_0))

adb_pred
