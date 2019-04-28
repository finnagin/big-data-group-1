library(ggplot2)
library(glmnet)
library(faraway)
library(tidyverse)

###############################################################

?fat
X <- as.matrix(fat[, 9:18])
y <- as.numeric(fat[, 1])

##### Lasso Estimates
lasso <- glmnet(X, y)
plot(lasso)

lasso.cv <- cv.glmnet(X, y)
lasso1 <- glmnet(X, y, lambda = lasso.cv$lambda.min)
lasso1$beta
lasso2 <- glmnet(X, y, lambda = lasso.cv$lambda.1se)
lasso2$beta

coef(lasso.cv)

##### Ridge Estimates
ridge <- glmnet(X, y, alpha = 0)
ridge.cv <- cv.glmnet(X, y, alpha = 0)
ridge1 <- glmnet(X, y, lambda = ridge.cv$lambda.min, alpha = 0)
ridge1$beta

coef(ridge.cv)

######################################################33

library(tidyverse)
library(GGally)

pop_data <- read_csv("psam_p41.csv")

#pop_data <- tbl_df(pop_data)
pop_new <- select(pop_data, AGEP, JWMNP)

#pop_new$PINCP <- as.numeric(pop_data$PINCP)
pop_new$WAGP <- as.numeric(pop_data$WAGP)
pop_new$JWMNP <- as.numeric(pop_data$JWMNP)

pop_new$JWAP <- as.numeric(pop_data$JWAP)
pop_new$AGEP <- as.numeric(pop_data$AGEP)
pop_new$SCHL <- as.numeric(pop_data$SCHL)
pop_new$WKHP <- as.numeric(pop_data$WKHP)
pop_new$PWGTP <- as.numeric(pop_data$PWGTP)
#pop_new$WKW <- as.numeric(pop_data$WKW)

cow1_f <- function(x){
  if(!is.na(x) && (x == 1 || x == 2)){
    return(1)
  } else {
    return(0)
  }
}
cow4_f <- function(x){
  if(!is.na(x) && (x == 8 || x == 9)){
    return(1)
  } else {
    return(0)
  }
}
cow2_f <- function(x){
  if(!is.na(x) && (x == 3 || x == 4 || x == 5)){
    return(1)
  } else {
    return(0)
  }
}
cow3_f <- function(x){
  if(!is.na(x) && (x == 6 || x == 7)){
    return(1)
  } else {
    return(0)
  }
}


cow_f <- function(x){
  if (cow1_f(x)){
    return (1)
  } else if(cow2_f(x)){
    return (2)
  } else if(cow3_f(x)){
    return (3)
  } else if(cow4_f(x)){
    return (4)
  } else {
    return (NA)
  }
}

jwtr1_f <- function(x){
  if(!is.na(x) && (x == 1 || x == 8)){
    return(1)
  } else {
    return(0)
  }
}
jwtr2_f <- function(x){
  if(!is.na(x) && (x >=2 && x <= 7)){
    return(1)
  } else {
    return(0)
  }
}
jwtr3_f <- function(x){
  if(!is.na(x) && (x == 9 || x == 10)){
    return(1)
  } else {
    return(0)
  }
}
jwtr4_f <- function(x){
  if(!is.na(x) && (x == 11 || x == 12)){
    return(1)
  } else {
    return(0)
  }
}

wkw1_f <- function(x){
  if(!is.na(x) && ( x >= 5)){
    return(1)
  } else {
    return(0)
  }
}
wkw2_f <- function(x){
  if(!is.na(x) && ( x == 4)){
    return(1)
  } else {
    return(0)
  }
}
wkw3_f <- function(x){
  if(!is.na(x) && (x <= 3 )){
    return(1)
  } else {
    return(0)
  }
}

mar_f <- function(x){
  if(!is.na(x) && (x == 1)){
    return(1)
  } else {
    return(0)
  }
}

mils_f <- function(x){
  if(!is.na(x) && (x == 1 || x == 2 || x == 3)){
    return(1)
  } else {
    return(0)
  }
}

sex_f <- function(x){
  if(!is.na(x) && (x == 1)){
    return(1)
  } else {
    return(0)
  }
}

insur_f <- function(x,y){
  if(!is.na(x) && (x == 2 && y == 2)){
    return(0)
  } else {
    return(1)
  }
}

pop_new$COW1 <- as.numeric(lapply(as.numeric(pop_data$COW), cow1_f))
pop_new$COW2 <- as.numeric(lapply(as.numeric(pop_data$COW), cow2_f))
pop_new$COW3 <- as.numeric(lapply(as.numeric(pop_data$COW), cow3_f))
pop_new$COW4 <- as.numeric(lapply(as.numeric(pop_data$COW), cow4_f))

pop_new$JWTR1 <- as.numeric(lapply(as.numeric(pop_data$JWTR), jwtr1_f))
pop_new$JWTR2 <- as.numeric(lapply(as.numeric(pop_data$JWTR), jwtr2_f))
pop_new$JWTR3 <- as.numeric(lapply(as.numeric(pop_data$JWTR), jwtr3_f))
pop_new$JWTR4 <- as.numeric(lapply(as.numeric(pop_data$JWTR), jwtr4_f))

pop_new$WKW1 <- as.numeric(lapply(as.numeric(pop_data$WKW), wkw1_f))
pop_new$WKW2 <- as.numeric(lapply(as.numeric(pop_data$WKW), wkw2_f))
pop_new$WKW3 <- as.numeric(lapply(as.numeric(pop_data$WKW), wkw3_f))

pop_new$MART <- as.numeric(lapply(as.numeric(pop_data$MAR), mar_f))
pop_new$MILT <- as.numeric(lapply(as.numeric(pop_data$MIL), mils_f))
pop_new$SEXT <- as.numeric(lapply(as.numeric(pop_data$SEX), sex_f))

pop_new$INSUR <- as.numeric(mapply(insur_f, pop_data$PRIVCOV, pop_data$PUBCOV))

#####################################################

pop_new_2 <- tbl_df(pop_new)
tracemem(pop_new) == tracemem(pop_new_2)

pop_new_2$COW1 <- NULL
pop_new_2$COW2 <- NULL
pop_new_2$COW3 <- NULL
pop_new_2$COW4 <- NULL

pop_new_2$JWTR2 <- NULL
pop_new_2$JWTR3 <- NULL
pop_new_2$JWTR4 <- NULL

pop_new_2$WKW1 <- NULL
pop_new_2$WKW2 <- NULL

pop_new_2$COW <- as.numeric(lapply(as.numeric(pop_data$COW),cow_f))

pop_omit_2 <- na.omit(pop_new_2)

y2 <- pop_omit_2$WKHP

pop_omit_2$WKHP <- NULL
pop_omit_2$COW1 <- NULL
pop_omit_2$COW2 <- NULL
pop_omit_2$COW3 <- NULL
pop_omit_2$COW4 <- NULL
pop_omit_2$COW <- NULL

X2 <- as.matrix(pop_omit_2)

lasso <- glmnet(X2, y2)
lasso.cv <- cv.glmnet(X2, y2)

coef(lasso.cv)


pop_lm_data_2 <- na.omit(pop_new_2)
pop_lm_2 <- lm(WKHP^2 ~ as.factor(COW)+JWMNP+WAGP+JWAP+as.factor(JWTR1)+as.factor(WKW3)+as.factor(MART)+as.factor(SEXT), 
               data = filter(pop_lm_data_2, WKHP < 50, WKHP > 10))
summary(pop_lm_2)

pop_lm_data_2$WKHP2 <- pop_lm_data_2$WKHP^2

ggpairs(sample_n(select(filter(pop_lm_data_2, WKHP < 50, WKHP > 10, JWMNP < 120, WAGP < 200000), WKHP2, WAGP, JWMNP, JWAP), 1000))

pop_lm_test <- lm(WKHP^2 ~ AGEP, data = filter(pop_lm_data_2, WKHP < 50, WKHP > 10))
par(mfrow = c(2, 2))
plot(pop_lm_test)


###################################################

pop_omit <- na.omit(pop_new)

y1 <- pop_omit$WAGP
j_rem <- pop_omit$JWMNP



pop_omit$WAGP <- NULL
pop_omit$JWMNP <- NULL

X <- as.matrix(pop_omit)

lasso <- glmnet(X, y1)
lasso.cv <- cv.glmnet(X, y1)

coef(lasso.cv)

pop_lm_data <- na.omit(pop_new)
pop_lm <- lm(log(WAGP) ~ JWMNP+AGEP+SCHL+WKHP+WKW3+MART+SEXT+INSUR, data = pop_lm_data)
summary(pop_lm)

vif(pop_lm)

plot(pop_lm)

pop_lm_data$LOGAGEP <- log(pop_lm_data$AGEP)
pop_lm_data$LOGWAGP <- log(pop_lm_data$WAGP)
pop_lm_data$LOGJWMNP <- log(pop_lm_data$JWMNP)
pop_lm_data$JWMNPSQ <- pop_lm_data$JWMNP^2

ggpairs(sample_n(filter(select(pop_lm_data, LOGWAGP, JWMNP, SCHL, AGEP, WKHP),LOGWAGP > log(2500), JWMNP < (120), SCHL > 9), 1000))
ggpairs(sample_n(filter(select(pop_lm_data, LOGWAGP, JWMNP, SCHL, LOGAGEP, WKHP),LOGWAGP > log(2500), JWMNP < (120), SCHL > 9), 1000))
  

############################################

pop_vet <- select(pop_data, AGEP, JWMNP)

#pop_new$PINCP <- as.numeric(pop_data$PINCP)
pop_vet$WAGP <- as.numeric(pop_data$WAGP)

pop_vet$AGEP <- as.numeric(pop_data$AGEP)

pop_vet$PWGTP <- as.numeric(pop_data$PWGTP)
pop_vet$SCHL <- as.numeric(pop_data$SCHL)

#pop_vet$DIS <- as.numeric(pop_data$DIS)

pop_vet$DRAT <- as.numeric(pop_data$DRAT)
pop_vet$VPS <- as.factor(pop_data$VPS)

na0 <- function(x){
  if (is.na(x)){
    return (0)
  } else {
    return (x)
  }
}

pop_vet$JWMNP <- as.numeric(lapply(as.numeric(pop_data$JWMNP), na0))
pop_vet$WKHP <- as.numeric(lapply(as.numeric(pop_data$WKHP), na0))

vet_omit <- na.omit(pop_vet)

y3 <- vet_omit$DRAT

vet_omit$DRAT <- NULL
vet_omit$VPS <- NULL

X3 <- as.matrix(vet_omit)

lasso <- glmnet(X3, y3)
lasso.cv <- cv.glmnet(X3, y3)

coef(lasso.cv)

# Gulf war erra
vps1_f <- function(x){
  if(!is.na(x) && (x <= 5)){
    return(1)
  } else {
    return(0)
  }
}
# WWII - Vietnam
vps2_f <- function(x){
  if(!is.na(x) && (x > 5 && x < 15 )){
    return(1)
  } else {
    return(0)
  }
}
# Pre WWII
vps3_f <- function(x){
  if(!is.na(x) && (x == 15)){
    return(1)
  } else {
    return(0)
  }
}

vps_f <- function(x){
  if (vps1_f(x)){
    return (1)
  } else if(vps2_f(x)){
    return (2)
  } else if(vps3_f(x)){
    return (3)
  } else {
    return (NA)
  }
}





#pop_vet$VPS1 <- as.numeric(lapply(as.numeric(pop_data$VPS), vps1_f))
#pop_vet$VPS2 <- as.numeric(lapply(as.numeric(pop_data$VPS), vps2_f))
#pop_vet$VPS3 <- as.numeric(lapply(as.numeric(pop_data$VPS), vps3_f))
pop_vet$VPS <- as.numeric(lapply(as.numeric(pop_data$VPS), vps_f))

vet_lm_data <- na.omit(pop_vet)

vet_lm <- lm(DRAT ~ as.factor(VPS) + AGEP + WKHP + SCHL + PWGTP + WAGP, data = filter(pop_vet, DRAT < 6))
summary(vet_lm)

par(mfrow = c(2, 2))
plot(vet_lm)

dim(pop_vet)
dim(pop_lm_data_2)
