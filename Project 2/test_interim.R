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

train_idx <- sample(1:nrow(X),11200)

oob.err<-double(11)
val.err<-double(11)

#mtry is no of Variables randomly chosen at each split
for(mtry in 5:15) 
{
  rf<-randomForest(y ~ . , data = X , subset = train_idx,mtry=mtry,ntree=500) 
  oob.err[mtry-4] <- rf$err.rate[500] #Error of all Trees fitted
  
  pred<-predict(rf,X[-train_idx,])
  val.err[mtry-4]<- with(X[-train_idx,],mean(y!=pred))
}

rf <- randomForest(y~.,data=X, ntrees=1500, mtry=14, subset=train_idx)

rf_pred<-predict(rf,X[-train_idx,])
rf.err<-with(X[-train_idx,],mean(y!=rf_pred))

svm_model <- svm(y~.,data=X, kernel="radial", subset=train_idx)

svm_pred<-predict(svm_model,X[-train_idx,])
svm.err<- with(X[-train_idx,],mean(y!=svm_pred))
