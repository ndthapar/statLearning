rm(list= ls())
library(verification)
setwd("/Users/neelanshuthapar/Desktop/crypto_csc687_final")
library(corrplot)
original = read.csv("./coin_Bitcoin.csv")
original = original
names(original)
dim(original)
sum(is.na(original))

###TO BE RUN ONLY ONCE!
tomorrowHigh = original$High
for(i in 1:length(tomorrowHigh)){
  tomorrowHigh[i] = tomorrowHigh[i+1]
}
original = data.frame(original, tomorrowHigh)

#drop new null vals
sum(is.na(original))
original = na.omit(original)

#drop zeroes
zero.index = which(original$Volume == 0)
print(zero.index)
original = original[-zero.index,]

summary(original)

#correlation initial for regression
corrReg = cor(original[5:11])
corrplot(corrReg, method="color")

#train test split
num_samples = dim(original)[1]
train_index = sample(num_samples, num_samples*0.8)
train = original[train_index,]
test = original[-train_index,]


#single variable at a time -- high
lm.fit.high = glm(tomorrowHigh ~ High, data=original, subset=train_index)
summary(glm.fit.high)
mean((test$tomorrowHigh-predict(glm.fit.high, test))[-train_index]^2)
cv.glm(data=original, glm(tomorrowHigh ~ High, data=original),K=5)$delta[1]
cv.glm(data=original, glm(tomorrowHigh ~ High, data=original),K=10)$delta[1]
cv.glm(data=original, glm(tomorrowHigh ~ High, data=original))$delta[1]


#single variable at a time -- low
lm.fit.low = glm(tomorrowHigh ~ Low, data=original, subset=train_index)
summary(glm.fit.high)
mean((test$tomorrowHigh-predict(lm.fit.low, test))[-train_index]^2)
cv.glm(data=original, glm(tomorrowHigh ~ Low, data=original),K=5)$delta[1]
cv.glm(data=original, glm(tomorrowHigh ~ Low, data=original),K=10)$delta[1]
cv.glm(data=original, glm(tomorrowHigh ~ Low, data=original))$delta[1]

#single variable at a time -- open
lm.fit.open = glm(tomorrowHigh ~ Open, data=original, subset=train_index)
summary(glm.fit.high)
mean((test$tomorrowHigh-predict(lm.fit.open, test))[-train_index]^2)
cv.glm(data=original, glm(tomorrowHigh ~ Open, data=original),K=5)$delta[1]
cv.glm(data=original, glm(tomorrowHigh ~ Open, data=original),K=10)$delta[1]
cv.glm(data=original, glm(tomorrowHigh ~ Open, data=original))$delta[1]

#single variable at a time -- close
lm.fit.close = glm(tomorrowHigh ~ Close, data=original, subset=train_index)
summary(lm.fit.close)
mean((test$tomorrowHigh-predict(lm.fit.close, test))[-train_index]^2)
cv.glm(data=original, glm(tomorrowHigh ~ Close, data=original),K=5)$delta[1]
cv.glm(data=original, glm(tomorrowHigh ~ Close, data=original),K=10)$delta[1]
cv.glm(data=original, glm(tomorrowHigh ~ Close, data=original))$delta[1]


#single variable at a time -- volume
lm.fit.volume = glm(tomorrowHigh ~ Volume, data=original, subset=train_index)
mean((test$tomorrowHigh-predict(lm.fit.volume, test))[-train_index]^2)
cv.glm(data=original, glm(tomorrowHigh ~ Volume, data=original),K=5)$delta[1]
cv.glm(data=original, glm(tomorrowHigh ~ Volume, data=original),K=10)$delta[1]
cv.glm(data=original, glm(tomorrowHigh ~ Volume, data=original))$delta[1]

#single variable at a time -- marketcap
lm.fit.marketcap = glm(tomorrowHigh ~ Marketcap, data=original, subset=train_index)
mean((test$tomorrowHigh-predict(lm.fit.marketcap, test))[-train_index]^2)
cv.glm(data=original, glm(tomorrowHigh ~ Marketcap, data=original),K=5)$delta[1]
cv.glm(data=original, glm(tomorrowHigh ~ Marketcap, data=original),K=10)$delta[1]
cv.glm(data=original, glm(tomorrowHigh ~ Marketcap, data=original))$delta[1]

#omitting non-numerical values -- date and serial number for transactions are not relevant predictors
glm.fit.mlr = glm(tomorrowHigh ~ High + Low + Open + Close + Volume + Marketcap, data=train)
mean((test$tomorrowHigh-predict(glm.fit.mlr, test))[-train_index]^2)
cv.glm(data=original, glm(tomorrowHigh ~ High + Low + Open + Close + Volume + Marketcap, data=original),K=5)$delta[1]
cv.glm(data=original, glm(tomorrowHigh ~ High + Low + Open + Close + Volume + Marketcap, data=original),K=10)$delta[1]
cv.glm(data=original, glm(tomorrowHigh ~ High + Low + Open + Close + Volume + Marketcap, data=original))$delta[1]



#collinearity check
vif(glm.fit.mlr)


#lm with poly
glm.poly.err = rep(0,5)
for (i in 1:10){
  glm.fit.poly=lm(tomorrowHigh ~ poly(High,i) + poly(Low,i) + poly(Open,i) + poly(Close,i) + poly(Volume,i) + poly(Marketcap,i), data=train)
  glm.poly.err[i] = mean((test$tomorrowHigh-predict(glm.fit.poly, test))[-train_index]^2)
}
print(glm.poly.err)

summary(glm.fit)
confint(glm.fit)


#new approach to solving the problem -- classification
#reinitialize
rm(list= ls())
set.seed(2)
original = read.csv("./coin_Bitcoin.csv")

#start by creating factor
#shift gain from t+1 to t
closeShift = original$Close
for(i in 1:length(closeShift)){
  closeShift[i] = closeShift[i+1]
}
closeShift[length(closeShift)]
gain = as.factor(ifelse((closeShift - original$Close)<=0, 0, 1))
original = data.frame(original, gain)

#drop new null vals
sum(is.na(original))
original = na.omit(original)

#drop zeroes
zero.index = which(original$Volume == 0)
print(zero.index)
original = original[-zero.index,]
dim(original)

#normalize data points using min-max normalization
for(j in 5:10){
  avg=mean(original[,j])
  mini=min(original[,j])
  maxi=max(original[,j])
  for(i in 1:length(original[,j])){
    original[i,j]= (original[i,j] -mini)/(maxi-mini)
  }
}


#reinitialize train and test split on new original
num_samples = dim(original)[1]
train_index = sample(num_samples, num_samples*0.8)
train = original[train_index,]
test = original[-train_index,]






#logistic regression validation set approach
glm.fits.logistic = glm(gain ~ High + Low + Open + Close + Volume + Marketcap, data=original,subset=train_index, family=binomial)
glm.probs = predict(glm.fits.logistic, test, type="response")
length(glm.probs)
glm.pred = rep(1, 524)
glm.pred[glm.probs < 0.5]= 0
length(glm.pred)
length(test)
table(glm.pred, test$gain)

#test error
mean(glm.pred != test$gain)


#CV approach
#cv-5,10,LOOCV fold
library(boot)
glm.fits.logistic = glm(gain ~ High + Low + Open + Close + Volume + Marketcap, data=original,family=binomial)
cv.glm.5 = cv.glm(original, glm.fits.logistic, K=5)
cv.glm.10 = cv.glm(original, glm.fits.logistic, K=10)
cv.glm.loocv = cv.glm(original, glm.fits.logistic)
cv.glm.5$delta[1]
cv.glm.10$delta[1]
cv.glm.loocv$delta[1]

#ROC plot for validation set logistic regression
library(ROCR)
pred = prediction(glm.probs, test$gain)
perf= performance(pred, "tpr","fpr")
par(mar=c(5,5,5,5))
plot(perf, colorize=TRUE)

#classification tree approach
library(tree)
?tree
tree.bitcoin = tree(gain ~  High + Low + Open + Close + Volume + Marketcap, data=original, subset=train_index)
summary(tree.bitcoin)
sum(original$gain == 1)
library(rpart)
tree.bit = rpart(gain ~  High + Low + Open + Close + Volume + Marketcap, data=original, method="class", control=(cp=0.0001))
summary(tree.bit)
par(mar=c(1,1,1,1))
plot(tree.bit)
text(tree.bit, pretty = 0, cex=0.75)

#correlation between predictors
library(corrplot)
corr = cor(original[5:10])
head(round(corr, 2))
corrplot(corr, method = "circle")

#randomForest approach to establish variable importance
library(randomForest)
set.seed(1)
rf.bitcoin = randomForest(gain ~ High + Low + Open + Close + Volume + Marketcap,data=original,subset=train_index,mtry=sqrt(6),ntree=100,importance =T)
yhat.rf=predict(rf.bitcoin ,newdata=test)
mean(yhat.rf != test$gain)
par(mar=c(3,4,2,4))
plot(importance(rf.bitcoin))
summary(rf.bitcoin)
rf.5 = rfcv(original[,5:10], original[,11],cv.fold=5)
rf.5$error.cv

rf.10 = rfcv(original[,5:10], original[,11],cv.fold=10)
rf.10$error.cv

rf.loocv = rfcv(original[,5:10], original[,11])
rf.loocv$error.cv
?rfev


#LDA
library(MASS)
?lda
?fix
lda.fit = lda(gain ~ Low + High + Open + Close + Volume + Marketcap, data=original, subset=train_index)
lda.pred=predict(lda.fit, newdata=test, type="response")
lda.pred
length(lda.pred$class)
lda.class=lda.pred$class
table(lda.class, test$gain)
mean(lda.class!=test$gain)






#QDA
qda.fit=qda(gain ~ High + Low + Open + Close + Volume + Marketcap, data=original, subset=train_index)
qda.fit
qda.class=predict(qda.fit, newdata=test, type="response")$class
table(qda.class, test$gain)
mean(qda.class!=test$gain)


#svm approach
library(e1071)
svmfit = svm(gain ~ High + Low + Open + Close + Volume + Marketcap, data=original, subset=train_index, kernel="radial", gamma=1, cost=1)
summary(svmfit)
svm.pred = predict(svmfit, newdata = test, type="response")
mean(svm.pred!=test$gain)

svmfit = svm(gain ~ High + Low + Open + Close + Volume + Marketcap, data=original, subset=train_index, kernel="linear", gamma=1, cost=1)
summary(svmfit)
svm.pred = predict(svmfit, newdata = test, type="response")
mean(svm.pred!=test$gain)

#linear svm with cv -- no tune.control is the default cv-10
tune.out = tune(svm, gain ~ High + Low + Open + Close + Volume + Marketcap, data=original,kernel="linear", ranges=list(cost=c(0.1,1,10), gamma=c(0,0.5,1)))
summary(tune.out)
tune.out = tune(svm, gain ~ High + Low + Open + Close + Volume + Marketcap, data=original,kernel="linear", ranges=list(cost=c(0.1,1,10), gamma=c(0,0.5,1)), tunecontrol=tune.control(cross=5))
summary(tune.out)
length(original)



#radial svm with cv
tune.out = tune(svm, gain ~ High + Low + Open + Close + Volume + Marketcap, data=original,kernel="radial", ranges=list(cost=c(0.1,1,10), gamma=c(0,0.5,1)))
summary(tune.out)
tune.out = tune(svm, gain ~ High + Low + Open + Close + Volume + Marketcap, data=original,kernel="radial", ranges=list(cost=c(0.1,1,10), gamma=c(0,0.5,1)), tunecontrol =tune.control(cross=5))
summary(tune.out)
tune.out = tune(svm, gain ~ High + Low + Open + Close + Volume + Marketcap, data=original,kernel="radial", ranges=list(cost=c(0.1,1,10), gamma=c(0,0.5,1)), tunecontrol =tune.control(cross=dim(original)[1]))
summary(tune.out)
length(original)




