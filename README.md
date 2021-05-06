# statLearning
Bitcoin MLR and logistic regression + related techniques


Purpose: to predict price value and price direction of BTC. Price value defined as the High price at t+1 predicted at time point t. Price direction defined as the difference between closing prices at t+1 and t.

NOTE: Linear regression and Classification based techniques were split. At line 118 the workspace is reset to pursue clasification related problems. 

Data: Cryptocurrency Historical Prices by SRK

https://www.kaggle.com/sudalairajkumar/cryptocurrencypricehistory

Methods applied and functions used in order:

Linear regression: glm()
Cross validation: cv.glm()
Logistic regression: glm()
Classification tree: tree()
Classification tree with modified control: rpart()
RandomForest: randomForest()
RandomForest cross validation: rfcv()
Linear Discriminant Analysis (LDA): lda()
Quadratic Discriminant Analysis (QDA): qda()
Support Vector Machine (SVM):

Sections within code are denoted by comments describing action being taken


libraries used:
-Verification
-corrplot
-boot
-ROCR
-tree
-rpart
-randomForest
-MASS
-E1071
