dataset=read.csv('Data.csv')

#missing data

# dataset$Age = ifelse(is.na(dataset$Age),
#                      ave(dataset$Age, FUN = function(x) mean(x, na.rm = TRUE)),
#                      dataset$Age)
# 
# dataset$Salary = ifelse(is.na(dataset$Salary),
#                      ave(dataset$Salary, FUN = function(x) mean(x, na.rm = TRUE)),
#                      dataset$Salary)

# Categorical Data: Unlike python we don't need the dummy variables. We can set the
# variables as factors and label those factors. Then the ML algorithm will know that 
# there isn't a relational order between the countries.

# dataset$Country = factor(dataset$Country,
#                          levels = c('France', 'Spain', 'Germany'),
#                          labels = c(1, 2, 3))
# 
# dataset$Purchased = factor(dataset$Purchased,
#                          levels = c('No', 'Yes'),
#                          labels = c(0, 1))

# Splitting train and test
library(caTools)

set.seed(123)

split = sample.split(dataset$Purchased, SplitRatio = 0.8)
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)

# Scaling 
# Factors don't count as numeric. Need to specify columns otherwise error.

# training_set[,2:3] = scale(training_set[,2:3])
# test_set[,2:3] = scale(test_set[,2:3])
