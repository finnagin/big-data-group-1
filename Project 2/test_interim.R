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
  
  hog <- HOG(im, cells = 3, orientations = 6)
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
X2 <- as_tibble(X,colnames=cn)
colnames(X2)=cn

rf <- randomForest(y~.,data=X2)