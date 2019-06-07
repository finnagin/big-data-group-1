# Project 2 Code
# Group1: Finn Womack, Hisham Jashami, and Rama Krishna Baisetti
# Cactus Classification


library(OpenImageR)
#library(tidyverse)

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

ans <- read.csv(file.path(file_dir,"train.csv"))
ans$id <- as.character(ans$id)

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

# Load test files

len<-length(features(train_path,train_files[1]))
cn<-character()
for(a in 1:len){
  cn <- append(cn,paste("x",as.character(a), sep = ""))
}
cn <- append(cn,"y")

XTest <- matrix(NA, nrow=length(test_files), ncol=len+1)
i <- 1
for(file in test_files){
  XTest[i,] <- append(features(test_path,file),ans$has_cactus[ans$id==file])
  i <- i+1
}
colnames(XTest)=cn
XTest <- as.data.frame(XTest)
XTest$y <- as.factor(XTest$y)

set.seed(Sys.time())
train_idx <- sample(1:nrow(X),11200)

#########################################################
#                      Week 1                           #
#########################################################

oob_err<-double(11)
val_err<-double(11)

#mtry is no of Variables randomly chosen at each split
for(mtry in 5:15){
  rf<-randomForest(y ~ . , data = X , subset = train_idx,mtry=mtry,ntree=500) 
  oob_err[mtry-4] <- rf$err.rate[500] #Error of all Trees fitted
  
  pred<-predict(rf,X[-train_idx,])
  val_err[mtry-4]<- with(X[-train_idx,],mean(y!=pred))
}
print(val_err)
# mtry was best at 14



val_err <- double(4)
i <- 1
for(k_type in c("linear", "sigmoid", "polynomial", "radial")){
  svmfit <- svm(y~.,data=X, kernel=k_type, subset=train_idx)
  pred<-predict(svmfit,X[-train_idx,])
  val_err[i]<- with(X[-train_idx,],mean(y!=pred))
  i = i+1
}
print(val_err)
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
rf_pred_1<-predict(rf,dplyr::filter(X[-train_idx,],y=="1"))
rf_pred_0<-predict(rf,dplyr::filter(X[-train_idx,],y=="0"))
rf_err<-with(X[-train_idx,],mean(y!=rf_pred))
rf_err_1<- with(dplyr::filter(X[-train_idx,],y=="1"),mean(y!=rf_pred_1))
rf_err_0<- with(dplyr::filter(X[-train_idx,],y=="0"),mean(y!=rf_pred_0))

svm_model <- svm(y~.,data=X, kernel="radial", cost=5, subset=train_idx)

svm_pred<-predict(svm_model,X[-train_idx,])
svm_pred_1<-predict(svm_model,dplyr::filter(X[-train_idx,],y=="1"))
svm_pred_0<-predict(svm_model,dplyr::filter(X[-train_idx,],y=="0"))
svm_err<- with(X[-train_idx,],mean(y!=svm_pred))
svm_err_1<- with(dplyr::filter(X[-train_idx,],y=="1"),mean(y!=svm_pred_1))
svm_err_0<- with(dplyr::filter(X[-train_idx,],y=="0"),mean(y!=svm_pred_0))

######################################################
#                      Week 2                        #
######################################################

pc <- prcomp(X[,!names(X) %in% c("y")])  # Eigen decomposition of Sample Covariance
#pc1 <- prcomp(select(X,-y), scale. = T)  # Eigen decomposition of Sample Covariance

summary(pc)






Xpc = as.data.frame(pc$x[, 1:55])
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

X2 <- as.data.frame(Xpc)
X2$y <- as.character(X2$y)

adb <- gbm(y~.,data=X2[train_idx,], distribution = "adaboost", n.trees = 500)
adb_pred<-predict(adb,X2[-train_idx,],n.trees=500, type = "response")
adb_pred_1<-predict(adb,dplyr::filter(X2[-train_idx,],y=="1"),n.trees=500, type = "response")
adb_pred_0<-predict(adb,dplyr::filter(X2[-train_idx,],y=="0"),n.trees=500, type = "response")






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



adb_err<- with(X2[-train_idx,],mean(y!=adb_pred))
adb_err_1<- with(dplyr::filter(X2[-train_idx,],y=="1"),mean(y!=adb_pred_1))
adb_err_0<- with(dplyr::filter(X2[-train_idx,],y=="0"),mean(y!=adb_pred_0))

svm_model <- svm(y~.,data=Xpc, kernel="radial", cost=5, subset=train_idx)

svm_pred<-predict(svm_model,Xpc[-train_idx,])
svm_pred_1<-predict(svm_model,dplyr::filter(Xpc[-train_idx,],y=="1"))
svm_pred_0<-predict(svm_model,dplyr::filter(Xpc[-train_idx,],y=="0"))
svm_err<- with(Xpc[-train_idx,],mean(y!=svm_pred))
svm_err_1<- with(dplyr::filter(Xpc[-train_idx,],y=="1"),mean(y!=svm_pred_1))
svm_err_0<- with(dplyr::filter(Xpc[-train_idx,],y=="0"),mean(y!=svm_pred_0))

rf <- randomForest(y~.,data=Xpc, ntrees=1500, mtry=14, subset=train_idx)



rf_pred<-predict(rf,Xpc[-train_idx,])
rf_pred_1<-predict(rf,dplyr::filter(Xpc[-train_idx,],y=="1"))
rf_pred_0<-predict(rf,dplyr::filter(Xpc[-train_idx,],y=="0"))
rf_err<-with(Xpc[-train_idx,],mean(y!=rf_pred))
rf_err_1<- with(dplyr::filter(Xpc[-train_idx,],y=="1"),mean(y!=rf_pred_1))
rf_err_0<- with(dplyr::filter(Xpc[-train_idx,],y=="0"),mean(y!=rf_pred_0))


#############################################################
#                           Test                            #
#############################################################



# generate Test features
TestMat <- XTest[,!names(XTest) %in% c("y")]
TestMat1 <- dplyr::filter(XTest, y=="1")[,!names(XTest) %in% c("y")]
TestMat0 <- dplyr::filter(XTest, y=="0")[,!names(XTest) %in% c("y")]




Xpc = as.data.frame(pc$x[, 1:55])
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

X2 <- as.data.frame(Xpc)
X2$y <- as.character(X2$y)

adb <- gbm(y~.,data=X2[train_idx,], distribution = "adaboost", n.trees = 500)
adb_pred<-predict(adb,X2[-train_idx,],n.trees=500, type = "response")
adb_pred_1<-predict(adb,dplyr::filter(X2[-train_idx,],y=="1"),n.trees=500, type = "response")
adb_pred_0<-predict(adb,dplyr::filter(X2[-train_idx,],y=="0"),n.trees=500, type = "response")

# generate test features
Testpc <- as.data.frame(predict(pc,TestMat))
Testpc$y <- as.character(XTest$y)


adb_test<-predict(adb,Testpc,n.trees=500, type = "response")
adb_test_1<-predict(adb,dplyr::filter(Testpc, y == "1"),n.trees=500, type = "response")
adb_test_0<-predict(adb,dplyr::filter(Testpc, y == "0"),n.trees=500, type = "response")


class_con <- function(x){
  if(x >= .5){
    return("1")
  }
  if(x < .5){
    return("0")
  }
}


adb_test <- lapply(adb_test, class_con)
adb_test_1 <- lapply(adb_test_1, class_con)
adb_test_0 <- lapply(adb_test_0, class_con)

err_adb<- with(XTest,mean(y!=adb_test))
err_adb_1<- with(dplyr::filter(XTest,y=="1"),mean(y!=adb_test_1))
err_adb_0<- with(dplyr::filter(XTest,y=="0"),mean(y!=adb_test_0))


rf <- randomForest(y~.,data=Xpc, ntrees=1500, mtry=14, subset=train_idx)


rf_test<-predict(rf,Testpc)
rf_test_1<-predict(rf,dplyr::filter(Testpc,y=="1"))
rf_test_0<-predict(rf,dplyr::filter(Testpc,y=="0"))

err_rf<-with(XTest,mean(y!=rf_test))
err_rf_1<- with(dplyr::filter(XTest,y=="1"),mean(y!=rf_test_1))
err_rf_0<- with(dplyr::filter(XTest,y=="0"),mean(y!=rf_test_0))

svm_model <- svm(y~.,data=X, kernel="radial", cost=5, subset=train_idx)

svm_test<-predict(svm_model,XTest)
svm_test_1<-predict(svm_model,dplyr::filter(XTest,y=="1"))
svm_test_0<-predict(svm_model,dplyr::filter(XTest,y=="0"))
err_svm<- with(XTest,mean(y!=svm_test))
err_svm_1<- with(dplyr::filter(XTest,y=="1"),mean(y!=svm_test_1))
err_svm_0<- with(dplyr::filter(XTest,y=="0"),mean(y!=svm_test_0))
