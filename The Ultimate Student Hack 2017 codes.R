#Loading required libraries.

library(data.table)
library(mlr)

##Setting working directory.

setwd("C:\\Users\\rishabh.parihar\\Downloads\\")

#Loading datasets.

train = fread("train_TUS_hack.csv" , stringsAsFactors = TRUE , na.strings = c("" , "missing" , "NA"))
test = fread("test_TUS_hack.csv" , stringsAsFactors = TRUE , na.strings = c("" , "missing" , "NA"))

#Creating a new variable log_Count.

train$log_Count = log(train$Count)
test$log_Count = test$Count

#Capping extreme values.

train$log_Count[train$log_Count < 3] = 4

#extracting year , month , Day of month , Hour of day from date variable for train dataset.

train$ID = as.character(train$ID)
train$Year = substr(train$ID , 1 , 4)
train$Month = substr(train$ID , 5, 6)
train$DayOfMonth = substr(train$ID , 7 , 8)
train$HourOfDay = substr(train$ID , 9 , 10)
train$date = ISOdate(train$Year , train$Month , train$DayOfMonth , hour = train$HourOfDay)
train = train[order(date) , ]

#extracting year , month , Day of month , Hour of day from date variable for test dataset.

test$ID = as.character(test$ID)
test$Year = substr(test$ID , 1 , 4)
test$Month = substr(test$ID , 5, 6)
test$DayOfMonth = substr(test$ID , 7 , 8)
test$HourOfDay = substr(test$ID , 9 , 10)
test$date = ISOdate(test$Year , test$Month , test$DayOfMonth , hour = test$HourOfDay)
test = test[order(date) , ]

#Combining the train and test datasets.

combin = rbind(train , test)

#Creating new variables from date variable.

combin$DayOfWeek = as.POSIXlt(combin$date)$wday 
combin$DayOfYear = as.POSIXlt(combin$date)$yday
combin$WeekOfYear = as.integer(format(as.Date(combin$date) , "%W"))
combin$HourOfDay = as.integer(combin$HourOfDay)
combin$DayOfMonth = as.integer(combin$DayOfMonth)
combin$Month = as.integer(combin$Month)
combin$Year = as.integer(combin$Year)
combin$Year = ifelse(combin$Year == 2011 , 0.5 , ifelse(combin$Year == 2012 , 1 , 1.5))

#Creating an Index variable.

combin$Index = seq(1 , nrow(combin) , 1)

#Creating flag variables capturing some information.

combin$WeekendFlag = ifelse((combin$DayOfWeek == 0) | (combin$DayOfWeek == 6) , 1 , 0)

combin$HourOfDayFlag = ifelse(combin$HourOfDay <= 5 , 1 , combin$HourOfDay)

combin$HourOfDayFlag = ifelse((combin$HourOfDayFlag > 5) & (combin$HourOfDayFlag <= 20), 2 , combin$HourOfDayFlag)

combin$HourOfDayFlag = ifelse(combin$HourOfDayFlag > 20 , 3 , combin$HourOfDayFlag)

#Extracting train and test datasets.

Train = combin[1:nrow(train) , ]
Test = combin[-(1:nrow(train)) , ]

#Subsetting columns.

Final_train = as.data.frame(Train[, c('Index' ,'DayOfYear' , 'DayOfMonth' , 'Month' , 'Year' , 'DayOfWeek' , 'HourOfDay' , 
                        'WeekendFlag' , 'WeekOfYear' , 'HourOfDayFlag' , 'log_Count')])


Final_test = as.data.frame(Test[, c('Index' ,'DayOfYear' , 'DayOfMonth' , 'Month' , 'Year' , 'DayOfWeek' , 'HourOfDay' , 
                      'WeekendFlag' , 'WeekOfYear' , 'HourOfDayFlag' , 'log_Count')])

##Creating training and test tasks.

train_task = makeRegrTask(data = Final_train ,  target = "log_Count")
test_task = makeRegrTask(data = Final_test , target = "log_Count")

#Creating a Gradient boosting learner to do the regression task.

regr_gbm_lrn = makeLearner("regr.gbm", predict.type = "response")

#specify tuning method

rancontrol_gbm <- makeTuneControlRandom(maxit = 100L)

#5 fold cross validation

set_cv_gbm <- makeResampleDesc("CV",iters = 5L)

#Specifying the parameter space for tuning.

gbm_par<- makeParamSet(
  makeDiscreteParam("distribution", values = "gaussian"),
  makeIntegerParam("n.trees", lower = 100, upper = 1000), #number of trees
  makeIntegerParam("interaction.depth", lower = 2, upper = 20), #depth of tree
  makeIntegerParam("n.minobsinnode", lower = 10, upper = 200),
  makeNumericParam("shrinkage",lower = 0.01, upper = 1)
)

#tuning the learner.

tune_gbm <- tuneParams(learner = regr_gbm_lrn, task = train_task , resampling = set_cv_gbm , measures = mse,
                       par.set = gbm_par, control = rancontrol_gbm)

#Setting the best hyperparameters(obtained after tuning) for the learner.

final_gbm <- setHyperPars(learner = regr_gbm_lrn, par.vals = tune_gbm$x)

#Training the learner on train_task.

train_regr_gbm = train(final_gbm , train_task)

#Predicting on test_task.

model_gbm_pred = predict(train_regr_gbm , test_task)

#Creating the submission file.

submit_gbm = data.frame(ID = test$ID , Count = exp(model_gbm_pred$data$response))
write.csv(submit_gbm , "gbm_7.csv" , row.names = FALSE)

##Creating a XgBoost learner to do the regression task.

regr_xgb_lrn = makeLearner("regr.xgboost", predict.type = "response")

#specify tuning method

rancontrol_xgb <- makeTuneControlRandom(maxit = 100L)

#5 fold cross validation

set_cv_xgb <- makeResampleDesc("CV",iters = 5L)

#Specifying the parameter space for tuning

xgb_par <- makeParamSet(
  makeIntegerParam("nrounds",lower=100,upper=1000),
  makeIntegerParam("max_depth",lower=3,upper=20),
  makeNumericParam("lambda",lower=0.55,upper=0.60),
  makeNumericParam("eta", lower = 0.001, upper = 0.5),
  makeNumericParam("subsample", lower = 0.10, upper = 0.80),
  makeNumericParam("min_child_weight",lower=1,upper=5),
  makeNumericParam("colsample_bytree",lower = 0.2,upper = 0.8)
)

tune_xgb <- tuneParams(learner = regr_xgb_lrn, task = train_task , resampling = set_cv_xgb , measures = rmse,
                       par.set = xgb_par, control = rancontrol_xgb)

#Setting the best hyperparameters(obtained after tuning) for the learner.

final_xgb <- setHyperPars(learner = regr_xgb_lrn, par.vals = tune_xgb$x)

#Training the learner on train_task.

train_xgb = train(final_xgb , train_task)

#Predicting on test_task.

model_xgb_pred = predict(train_xgb , test_task)

#Creating the submission file.

submit_xgb = data.frame(ID = test$ID , Count = exp(model_xgb_pred$data$response))
write.csv(submit_xgb , "xgb_2.csv" , row.names = FALSE)

#Creating the final submission file containing the weighted average of XgBoost Predictions and GBM predictions.

submit_gbm_xgb = data.frame(ID = test$ID , Count = 0.3*submit_gbm$Count + 0.7*submit_xgb$Count)
write.csv(submit_gbm_xgb , "xgb_gbm_7.csv" , row.names = FALSE)




















