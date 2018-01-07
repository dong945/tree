# ==============================================================================
# 機器學習:決策樹
# 資料集:https://archive.ics.uci.edu/ml/datasets/statlog+(german+credit+data)
# ==============================================================================
#
# 先用iris資料集來展示決策樹及說明如何裁剪節點
library(tree)
iristree <- tree(Species ~ ., data = iris)
iristree    # 列出決策樹的條件
class(iristree)
plot(iristree)
text(iristree)
## 裁剪節點: snip.tree
# 7) Petal.Width > 1.75 (virginica, virginica) 
# 12) Petal.Length < 4.95 (versicolor, versicolor)
plot(irissnip, col='blue')
irissnip <- snip.tree(iristree, nodes=c(7,12)) 
text(irissnip, digits=2)
# 顯示原有條件及裁剪後
iristree
irissnip
# 分類迴歸樹
library(rpart) # 'rpart' 
iristree <- rpart(Species ~ ., data = iris)
iristree
class(iristree)
library(rpart.plot)
library(rattle) # Fancy tree plot
fancyRpartPlot(iristree)
# =================================================================
# 信用風險預測, 使用C5.0
# 資料集有17 variables, 
# 第17個Variable(default)為Level標籤, no (無違約) yes (會違約)
# =================================================================
credit <- read.csv(file.choose()) # select credit.csv
str(credit)
# 分別取支票餘額(checking_balance)、存款餘額(savings_balance)與default
# 建交叉列聯表
prop.table(table(credit$checking_balance, credit$default))
prop.table(table(credit$savings_balance, credit$default))
# 由結果來看, 餘額越高越不容易違約
# 看貸款期限及金額的分佈情形
summary(credit$months_loan_duration)
boxplot(credit$months_loan_duration)
summary(credit$amount)
boxplot(credit$amount)
# 計算default
table(credit$default)
# 建立一個隨機樣本, 供training and test
# 使用set.seed, 固定亂數種子
set.seed(12345) 
credit_rand <- credit[order(runif(1000)), ] # 將原有的資料集打散重排列
# 檢查打亂後的資料集, 統計值是否會改變
summary(credit$amount) 
summary(credit_rand$amount)
head(credit$amount) 
head(credit_rand$amount)
# 另一種取樣方式, 用sample
# 900筆當train Data, 100筆當test data
?sample
idx <- sample(1:1000, 900)
credit_train <- credit[idx,]   ## 訓練資料集
credit_test <- credit[-idx,]   ## 測試資料集
#####
credit_train <- credit_rand[1:900, ] 
credit_test  <- credit_rand[901:1000, ]
# 檢查train及test data, 統計值是否差不多
prop.table(table(credit_train$default))   
prop.table(table(credit_test$default))
# ================================================
library(C50) 
# 先用預設值, 建立C5.0決策樹
credit_model <- C5.0(credit_train[-17], credit_train$default) 
credit_model 
# 設定條件, 節點向下小於10, 即不再向下長
credit_model <- C5.0(credit_train[-17], credit_train$default, control=C5.0Control(winnow=T,minCases=10)) 
credit_model

summary(credit_model)
# 將test data放入模型中, 預測結果
credit_pred <- predict(credit_model, credit_test)
credit_pred
# 將預測結果及實際LEVEL, 做交叉列聯表
table(credit_pred, credit_test$default)

library(gmodels) # Various R programming tools for model fitting
CrossTable(credit_test$default, credit_pred, prop.chisq = FALSE, prop.c = FALSE, prop.r = FALSE, dnn = c('actual default', 'predicted default')) # Cross Tabulation with Tests for Factor Independence
# 提升決策樹的正確性(trials=10)
credit_boost10 <- C5.0(credit_train[-17], credit_train$default, trials = 10) # boosted decision tree with 10 trials
credit_boost10
summary(credit_boost10)
credit_boost_pred10 <- predict(credit_boost10, credit_test)
CrossTable(credit_test$default, credit_boost_pred10, prop.chisq = FALSE, prop.c = FALSE, prop.r = FALSE, dnn = c('actual default', 'predicted default')) # error rate is 23%
# 提升決策樹的正確性(trials=100)
credit_boost100 <- C5.0(credit_train[-17], credit_train$default, trials = 100) # boosted decision tree with 100 trials
credit_boost100
summary(credit_boost100)
credit_boost_pred100 <- predict(credit_boost100, credit_test)
CrossTable(credit_test$default, credit_boost_pred100, prop.chisq = FALSE, prop.c = FALSE, prop.r = FALSE, dnn = c('actual default', 'predicted default')) # error rate is 22%
# 放款違約是錯失機會成本的四倍, 建立成本矩陣
error_cost <- matrix(c(0, 1, 4, 0), nrow = 2)
error_cost 
# 加入成本矩陣, 為了減少違約(真實)被誤判為正常(預測)的部份
credit_cost <- C5.0(credit_train[-17], credit_train$default, costs = error_cost) 
credit_cost
summary(credit_cost)
credit_cost_pred <- predict(credit_cost, credit_test)
CrossTable(credit_test$default, credit_cost_pred, prop.chisq = FALSE, prop.c = FALSE, prop.r = FALSE, dnn = c('actual default', 'predicted default'))

