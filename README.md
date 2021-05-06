# statLearning
Bitcoin MLR and logistic regression + related techniques


Purpose: to predict price value and price direction of BTC. Price value defined as the High price at t+1 predicted at time point t. Price direction defined as the difference between closing prices at t+1 and t.

NOTE: Linear regression and Classification based techniques were split. At line 161 the workspace is reset to pursue clasification related techniques

Data: Cryptocurrency Historical Prices by SRK

https://www.kaggle.com/sudalairajkumar/cryptocurrencypricehistory

Methods applied and functions used in order:

Linear regression: glm()<br/>
Cross validation: cv.glm()<br/>
Logistic regression: glm()<br/>
Classification tree: tree()<br/>
Classification tree with modified control: rpart()<br/>
RandomForest: randomForest()<br/>
RandomForest cross validation: rfcv()<br/>
Linear Discriminant Analysis (LDA): lda()<br/>
Quadratic Discriminant Analysis (QDA): qda()<br/>
Support Vector Machine (SVM): tune()<br/>
SVM cross validation: tune()<br/>

Sections within code are denoted by comments describing action being taken


libraries used:
-Verification<br/>
-corrplot<br/>
-boot<br/>
-ROCR<br/>
-tree<br/>
-rpart<br/>
-randomForest<br/>
-MASS<br/>
-E1071<br/>
