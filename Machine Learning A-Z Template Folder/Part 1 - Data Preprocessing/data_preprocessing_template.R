#data preprocessing

#importing dataset

dataset = read.csv('Data.csv')

#taking care of missing values
dataset$Age = ifelse(is.na(dataset$Age),
                     ave(
                       dataset$Age,
                       FUN = function(x)
                         mean(x, na.rm = TRUE)
                     ),
                     dataset$Age)
dataset$Salary = ifelse(is.na(dataset$Salary),
                        ave(
                          dataset$Salary,
                          FUN = function(x)
                            mean(x, na.rm = TRUE)
                        ),
                        dataset$Salary)
#encoding categorical data

dataset$Country=factor(dataset$Country,levels=c('France','Spain','Germany'),labels = c(2,1,3))
dataset$Purchased=factor(dataset$Purchased,levels=c('Yes','No'),labels = c(0,1))

#splitting dataset into train and test sets

#install.packages('caTools')
library(caTools)
set.seed(123)
split=sample.split(dataset$Purchased,SplitRatio = 0.8)
training_set=subset(dataset,split==TRUE)
test_set=subset(dataset,split==FALSE)

#feature scaling

training_set=scale(training_set[,2:3])
test_set=scale(test_set[,2:3])




