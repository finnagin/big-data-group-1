library(OpenImageR)
library(tidyverse)
library(e1071)
library(randomForest)


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
X <- as_tibble(X,colnames=cn)
X <- X %>%
  mutate(y = factor(y))

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
