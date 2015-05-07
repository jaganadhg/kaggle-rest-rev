library(caret)

training_data = read.csv("Trainning_data.csv")
test_data = read.csv("Test_Data.csv")

sweep_params = trainControl(method = "cv",repeats = 5,allowParallel=TRUE)

nnet_model = train(revenue ~ . , data=training_data,method='gbm',trControl = sweep_params)
str(nnet_model)
save(nnet_model,file="RandomForest.rdata")


load("RandomForest.rdata")
nnet_model
