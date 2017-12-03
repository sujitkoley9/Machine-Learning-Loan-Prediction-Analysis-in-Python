#------------------Step 1: Importing library-------------------------------------------------
  library(ggplot2)
  library(dplyr)
  library(mlr)
  library(caret)
  library(ROCR)
  library(parallel)
  library(parallelMap) 
  library(dummies)
  library(corrplot)
  library(randomForest)
  library(FSelector)
  library(xgboost)


#-----------------Step 2: Read CSV file and prepare data quality report---------------------
  #Read CSV file
  setwd("W:\\Kaggle compettion\\Loan Prediction")
  Train_df <- read.csv("train.csv",header=T,as.is =T,na.strings ="")
  Test_df  <- read.csv("Test.csv",header=T,as.is =T,na.strings ="")
  Test_df$Loan_Status <-"Y"
  
  LoanRecords<-rbind(Train_df,Test_df)
  View(LoanRecords)


  #Data report generation
  summary(LoanRecords)
  Varible <-1
  Maximum  <-1
  Minimum <-1
  Data_type <-1
  StandardDeviation<-1
  Uniques <-1
  
  list <-names(LoanRecords)
  for(i in 1:length(list))
  {
    Column <- LoanRecords[,list[i]]
    Varible[i]<-list[i]
    Data_type[i] <- class(Column)
    Maximum[i] <-ifelse( Data_type[i]=="integer"| Data_type[i]=="numeric",max(Column,na.rm = T),0)
    Minimum[i] <-ifelse( Data_type[i]=="integer"| Data_type[i]=="numeric",min(Column,na.rm = T),0)
    StandardDeviation[i] <- ifelse( Data_type[i]=="integer"| Data_type[i]=="numeric",sd(Column,na.rm = T),0)
    Uniques[i]<- ifelse( Data_type[i]=="character",nlevels(as.factor(Column)),0)
  
    
  }
 
  missing_count <-colSums(is.na(LoanRecords))
 
  Summary_report <-data.frame(
                               Varible =Varible,
                               Data_type=Data_type,
                               Maximum=Maximum,
                               Minimum=Minimum,
                               StandardDeviation=StandardDeviation,
                               Unique=Uniques,
                               missing_count=missing_count
                               
                              )
  
  write.csv(Summary_report,file="DataReport.csv",row.names =F)
  
  
  
#-----------------------------Step 3:Missing Value Treatment--------------------------------------#
  
   # Treatment for Gender
  LoanRecords[which(is.na(LoanRecords$Gender)),"Gender"] <- "Missing"
  
   # Treatment for Married
  LoanRecords[which(is.na(LoanRecords$Married)),"Married"] <- "Missing"

   # Treatment for Self_Employed
  LoanRecords[which(is.na(LoanRecords$Self_Employed)),"Self_Employed"] <- "Missing"
  
   # Treatment for Credit_History
  data <-LoanRecords %>%group_by(Credit_History,Loan_Status)%>%summarise(n=n())%>%filter(Loan_Status=="Y")
  data1<-LoanRecords %>%group_by(Credit_History)%>%summarise(n=n())%>%filter(Credit_History %in% data$Credit_History )
  Loan_percen <- round(data$n/data1$n,4)
  summary_result <-data.frame(Credit_History =data$Credit_History,
                              Loan_status=data$Loan_Status,
                              Loan_percen=Loan_percen,
                              data_n=data$n,
                              data1_n=data1$n,
                              variable=rep("Credit_History",length(data$n))
  )
  LoanRecords[which(is.na(LoanRecords$Credit_History)),"Credit_History"] <- 1
  
   #6. Treatment for Dependents
   LoanRecords[which(is.na(LoanRecords$Dependents)),"Dependents"] <- "Missing"
   
   #6.Treatment for LoanAmount
   
   data <-LoanRecords %>%mutate(qnt=ntile(LoanAmount,10))%>%
          group_by(qnt,Loan_Status)%>%summarise(n=n())%>%filter(Loan_Status=="Y")
   data1<-LoanRecords %>%mutate(qnt=ntile(LoanAmount,10))%>%
          group_by(qnt)%>%summarise(n=n())%>%filter(qnt %in% data$qnt )
   
   Loan_percen <- round(data$n/data1$n,4)
   summary_result <-data.frame(qnt =data$qnt,
                               Loan_status=data$Loan_Status,
                               Loan_percen=Loan_percen,
                               data_n=data$n,
                               data1_n=data1$n,
                               variable=rep("LoanAmount",length(data$n))
   )
   
   #According to distribution "NA" beahave not any qnantile, so NA can be replaced with mean
   LoanRecords[which(is.na(LoanRecords$LoanAmount)),"LoanAmount"] <- mean(LoanRecords$LoanAmount,na.rm=T)
   
   #7.Loan_Amount_Term

   LoanRecords[which(is.na(LoanRecords$Loan_Amount_Term)),"Loan_Amount_Term"] <- mean(LoanRecords$Loan_Amount_Term,na.rm=T)
   

#--------------------------Step 4: Outlier Treatment----------------------------------------------#
   # Outlier treatment for only integer and nummeric variable
   # here 1.ApplicantIncome 2.CoapplicantIncome 3.LoanAmount 4.Loan_Amount_Term 
   
   List_var <- c("ApplicantIncome","CoapplicantIncome","LoanAmount")
   
   par(mfrow=c(3,2))
   
   for(i in 1:length(List_var))
   {
     boxplot(LoanRecords[,List_var[i]],main=List_var[i])
     
   }
   
   dev.off()
   for(i in 1:length(List_var))
   {
     x <- boxplot(LoanRecords[,List_var[i]],main=List_var[i])
     LoanRecords[which(LoanRecords[,List_var[i]] %in% x$out),List_var[i]] <- 
       mean(LoanRecords[,List_var[i]],na.rm = T)
     
   }
   
   
   
   
#---------------------------Step 5:Data Exploration & Feature engineering--------------------------------##
   
   #1.Dependents vs Loan_Status
    p <- ggplot(LoanRecords,aes(Dependents,fill=Loan_Status))
    p+geom_bar()+labs(x="Dependents",y="count",title="Dependents vs Loan_Status")
    
    #1.Dependents vs Loan_Status vs Gender
    
    p <- ggplot(LoanRecords,aes(Dependents,fill=Loan_Status))
    p+geom_bar()+facet_grid(.~Gender)+labs(x="Dependents",y="count",title="Dependents vs Loan_Status")
    
    
    #Creating dummy variable for Geneder
    Gender_dummies <- dummy(LoanRecords$Gender, sep = "_")
    Gender_dummies <- as.data.frame(Gender_dummies)
    Gender_dummies <- Gender_dummies[-1]
    LoanRecords    <- cbind(LoanRecords,Gender_dummies)
    
    #Creating dummy variable for Married
    Married_dummies <- dummy(LoanRecords$Married, sep = "_")
    Married_dummies <- as.data.frame(Married_dummies)
    Married_dummies <- Married_dummies[-1]
    LoanRecords    <- cbind(LoanRecords,Married_dummies)
    
    
    #Creating dummy variable for Self_Employed
    Self_Employed_dummies <- dummy(LoanRecords$Self_Employed, sep = "_")
    Self_Employed_dummies <- as.data.frame(Self_Employed_dummies)
    Self_Employed_dummies <- Self_Employed_dummies[-1]
    LoanRecords           <- cbind(LoanRecords,Self_Employed_dummies)
    
    
    #Creating dummy variable for Dependents
    Dependents_dummies <- dummy(LoanRecords$Dependents, sep = "_")
    Dependents_dummies <- as.data.frame(Dependents_dummies)
    Dependents_dummies <- Dependents_dummies[-1]
    LoanRecords        <- cbind(LoanRecords,Dependents_dummies)
    
    #Creating dummy variable for Education
    Education_dummies <- dummy(LoanRecords$Education, sep = "_")
    Education_dummies <- as.data.frame(Education_dummies)
    Education_dummies <- Education_dummies[-1]
    LoanRecords       <- cbind(LoanRecords,Education_dummies)
    
    
    #Creating dummy variable for Property_Area
    Property_Area_dummies <- dummy(LoanRecords$Property_Area, sep = "_")
    Property_Area_dummies <- as.data.frame(Property_Area_dummies)
    Property_Area_dummies <- Property_Area_dummies[-1]
    LoanRecords           <- cbind(LoanRecords,Property_Area_dummies)
    
    
   
    #1.Create New Variable TotalIncome
    LoanRecords$TotalIncome <-LoanRecords$ApplicantIncome + LoanRecords$CoapplicantIncome
    
    #Convert Loan_Status to factor
    LoanRecords$Loan_Status <- ifelse(LoanRecords$Loan_Status =='Y',1,0)
    LoanRecords$Loan_Status <-as.factor(LoanRecords$Loan_Status)
    
    
#-------------------Step 6: Calculating Multicollinearity and droping dependent variable---------
    LoanRecords_1 <- LoanRecords%>%select(-Loan_Status)
    
    #Identifying numeric variables
    numericData <- LoanRecords_1[sapply(LoanRecords_1, is.numeric)]
    
    #Calculating Correlation
    descrCor <- cor(numericData)
    
    # Checking Variables that are highly correlated
   
    highlyCorrelated = findCorrelation(descrCor, cutoff=0.7)
    
    #Identifying Variable Names of Highly Correlated Variables
    #highlyCorCol = names(numericData[,highlyCorrelated])
    highlyCorCol = names(numericData)[highlyCorrelated]
    
    #Remove highly correlated variables if present and create a new dataset
    LoanRecords <-LoanRecords[,-which(names(LoanRecords) %in% highlyCorCol)]
    # Visualize Correlation Matrix
    corrplot(descrCor, order = "FPC", method = "color", type = "lower",
             tl.cex = 0.7, tl.col = rgb(0, 0, 0))
    

    #---------------------------------- Droping unnecessary columns---------------
    
    LoanRecords <- LoanRecords %>% select(-c(Dependents,Education,Gender,Loan_ID,
                                             Married,Property_Area,Self_Employed,
                                             ApplicantIncome,CoapplicantIncome)) 
    
    names(LoanRecords)
#----------------------Step 7: Separating train/test dataset and Normalize data-------------   
    Train <- LoanRecords[1:614,]
    Test <- LoanRecords[615:981,]
    
    X_train <- Train %>% select(-Loan_Status)
    Y_train <- Train %>% select(Loan_Status)
    
    X_test  <- Test %>% select(-Loan_Status)
    Y_test  <- Test %>% select(Loan_Status)
  
    #-----Scaling
    X_train <- scale(X_train,center=TRUE,scale=TRUE)
    X_test  <- scale(X_test,center=TRUE,scale=TRUE)
  
    
    
    #--------PCA Analysis
    prin_comp <- prcomp(X_train, scale. = F)
    pr_var <- (prin_comp$sdev)^2
    prop_varex <- pr_var/sum(pr_var)
    
    #Cumulative Variance explains
    Cum_prop_varex<-cumsum(prop_varex)
    plot(Cum_prop_varex, xlab = "Principal Component",
         ylab = "Cumulative Proportion of Variance Explained",
         type = "b")
    
    # As per analysis, we can skip 2 principal componet, use only 13 components
    X_train <-prin_comp$x[,1:13] 
 
    X_test_predict  <-predict(prin_comp, newdata = X_test)
    X_test_predict <- as.data.frame(X_test_predict)
    X_test <-X_test_predict[,1:13]
   
    

#---------------------------Step 8: Run Algorithm-------------------------------------------------
    
    Train_data <- data.frame(X_train,"Loan_Status" =Y_train)
    Test_data  <- data.frame(X_test ,"Loan_Status" =Y_test)
  
    TrainTask <- makeClassifTask(data=Train_data,target ="Loan_Status",positive =1)
    TestTask  <- makeClassifTask(data=Test_data,target ="Loan_Status",positive =1)
    
 #---------------To Check importance of variable---------------------------------------------#
    
   
    im_feat <- generateFilterValuesData(TrainTask, method = c("information.gain","chi.squared"
                                                              ))
    
    plotFilterValues(im_feat,n.show = 20)
   
 #1.Logistic regression:  
    logistic.learner <- makeLearner("classif.logreg",predict.type ="response")
    
    #cross validation (cv) accuracy
     cv.logistic <- crossval(learner = logistic.learner,task = TrainTask,iters = 3,
                             stratify = TRUE,measures = acc,show.info = T)
     
    #cross validation accuracy
     cv.logistic$aggr
     
     #train model
   
     LogisticsModel <-mlr::train(learner=logistic.learner,task=TrainTask)
     getLearnerModel(LogisticsModel)
     
     
   #2.Decision Tree 
      DecisionTree.learner <- makeLearner("classif.rpart", predict.type = "response")   
      getParamSet("classif.rpart")
     #set 3 fold cross validation
      set_cv <- makeResampleDesc("CV",iters = 3L)
      #Search for hyperparameters
      DecisionTree.Parameter <- makeParamSet(
        makeIntegerParam("minsplit",lower = 2, upper = 10),
        makeIntegerParam("minbucket", lower = 2, upper = 10),
        makeIntegerParam("maxdepth", lower = 3, upper = 10),
        makeNumericParam("cp", lower = 0.001, upper = 0.2)
      )
    
      #search strategy
      ctrl <- makeTuneControlRandom(maxit = 200L)
      
      parallelStartSocket(cpus = detectCores())
      
      DecisionTree.tune <- tuneParams(learner = DecisionTree.learner, task = TrainTask, 
                                      resampling = set_cv,measures = acc, par.set = DecisionTree.Parameter, 
                                      control = ctrl, show.info = T)
    
    
      parallelStop()
    
      #set hyperparameters
      DecisionTree.learner.tune <- setHyperPars(DecisionTree.learner,par.vals = DecisionTree.tune$x)
      
      #cross validation (cv) accuracy
      cv.logistic <- crossval(learner = DecisionTree.learner.tune,task = TrainTask,iters = 3,
                              stratify = TRUE,measures = acc,show.info = T)
      
      #cross validation accuracy
      cv.logistic$aggr
      
      #train model
      Model_DecisionTree <- mlr::train(learner = DecisionTree.learner.tune,task = TrainTask)
      
      summary(getLearnerModel(Model_DecisionTree))
     
    #3.Random Forest
      RandomForest.learner <- makeLearner("classif.randomForest", predict.type = "response")  
      getParamSet("classif.randomForest")
      RandomForest.learner$par.vals <- list(
        importance = TRUE
      )
      #set 3 fold cross validation
      set_cv <- makeResampleDesc("CV",iters = 3L)
      
      #Search for hyperparameters
      RandomForest.Parameter <- makeParamSet(
        makeIntegerParam("ntree",lower = 50, upper = 100),
        makeIntegerParam("mtry", lower = 2, upper = 10),
        makeIntegerParam("nodesize", lower = 1, upper = 10)
      )
      
      #search strategy
      ctrl <- makeTuneControlRandom(maxit = 200L)
      
      parallelStartSocket(cpus = detectCores())
      
      RandomForest.tune <- tuneParams(learner = RandomForest.learner, task = TrainTask, 
                                      resampling = set_cv,measures = acc, par.set = RandomForest.Parameter, 
                                      control = ctrl, show.info = T)
      
      
      parallelStop()
      
      #set hyperparameters
      RandomForest.learner.tune <- setHyperPars(RandomForest.learner,par.vals = RandomForest.tune$x)
      
      #cross validation (cv) accuracy
      cv.logistic <- crossval(learner = RandomForest.learner.tune,task = TrainTask,iters = 3,
                              stratify = TRUE,measures = acc,show.info = T)
      
      #cross validation accuracy
      cv.logistic$aggr
      #train model
      Model_RandomForest <- mlr::train(learner = RandomForest.learner,task = TrainTask)
      summary(getLearnerModel(Model_RandomForest))
     
    
    #4.Xgboost:
      Xgboost.learner <- makeLearner("classif.xgboost", predict.type = "response")  
      
      Xgboost.learner$par.vals <- list( objective="binary:logistic", 
                                        eval_metric="error", nrounds=100L, eta=0.1)
      #set 3 fold cross validation
      set_cv <- makeResampleDesc("CV",iters = 3L)
      
      #Search for hyperparameters
      Xgboost.Parameter <- makeParamSet(
        makeNumericParam("lambda",lower=0.55,upper=0.60),
        makeIntegerParam("max_depth",lower = 3L,upper = 10L), 
        makeNumericParam("min_child_weight",lower = 1L,upper = 10L),
        makeNumericParam("subsample",lower = 0.5,upper = 1),
        makeNumericParam("colsample_bytree",lower = 0.5,upper = 1)
      )
      
      #search strategy
      ctrl <- makeTuneControlRandom(maxit = 200L)
      
      
      parallelStartSocket(cpus = detectCores())
      
      Xgboost.tune <- tuneParams(learner = Xgboost.learner, task = TrainTask, 
                                      resampling = set_cv,measures = acc, par.set = Xgboost.Parameter, 
                                      control = ctrl, show.info = T)
      
      
      parallelStop()
      
      #set hyperparameters
      Xgboost.learner.tune <- setHyperPars(Xgboost.learner,par.vals = Xgboost.tune$x)
      
      #cross validation (cv) accuracy
      cv.logistic <- crossval(learner = Xgboost.learner.tune,task = TrainTask,iters = 3,
                              stratify = TRUE,measures = acc,show.info = T)
      
      #cross validation accuracy
      cv.logistic$aggr
      
      #train model
      Model_Xgboost <- mlr::train(learner = Xgboost.learner.tune,task = TrainTask)
      summary(getLearnerModel(Model_Xgboost))
      
    
#-----------------------------------Ensembling Different Model-----------------------------#
      
      
 # Prdiction of Test task:
    Predict_Xgboost <- predict(Model_Xgboost, TestTask)
    Predict_RandomForest <- predict(Model_RandomForest, TestTask)
    Predict_DecisionTree <- predict(Model_DecisionTree, TestTask)
    Model_Logistics <- predict(LogisticsModel, TestTask)
    
    
    
    Model_prediction_avg <-  ((as.numeric(Model_Logistics$data$response)-1)+
                              (as.numeric(Predict_DecisionTree$data$response)-1)+
                              (as.numeric(Predict_RandomForest$data$response)-1)+
                              (as.numeric(Predict_Xgboost$data$response)-1)
                                 )/4
    
    
    Model_prediction_avg_10 <-ifelse(Model_prediction_avg>.5,1,0)
    
    
    Df<-data.frame(Loan_ID=Test_df$Loan_ID,
                   Loan_Status=Model_prediction_avg_10
    )
    write.csv(Df,file="Report.csv",row.names = F)