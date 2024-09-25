# Load the necessary libraries
library(caTools)
library(class)
library(e1071)
library(ggplot2)
library(pROC)
library(sets)
library(nnet)
library(caret)

# Load the dataset
dataset <- read.csv('Stress-Lysis.csv')

# Check for null values
p <- sum(is.na(dataset))

# Encoding the target feature as factor
dataset$Stress.Level <- factor(dataset$Stress.Level, levels = c(0, 1, 2))

# Function to remove outliers based on IQR
remove_outliers_iqr <- function(x, k = 1.5) {
  q <- quantile(x, probs = c(0.25, 0.75))
  iqr <- q[2] - q[1]
  lower_bound <- q[1] - k * iqr
  upper_bound <- q[2] + k * iqr
  x[which(x < lower_bound | x > upper_bound)] <- NA
  return(x)
}

# Define columns to normalize
columns_to_normalize <- c("Humidity", "Temperature", "Step.count")

# Data normalization function
normalize_data <- function(data, columns_to_normalize, scaling_parameters) {
  for (col_name in columns_to_normalize) {
    col_data <- data[[col_name]]
    
    # Use scale function directly
    normalized_column <- scale(col_data, center = scaling_parameters[[col_name]]$mean, scale = scaling_parameters[[col_name]]$sd)
    
    data[[col_name]] <- normalized_column
  }
  return(data)
}

# Divide into train and test data
set.seed(123)
split <- sample.split(dataset$Stress.Level, SplitRatio = 0.75)
training_set <- subset(dataset, split == TRUE)
test_set <- subset(dataset, split == FALSE)

# Save scaling parameters for normalization
scaling_parameters <- list()
for (col_name in columns_to_normalize) {
  scaling_parameters[[col_name]] <- list(mean = mean(training_set[[col_name]]), sd = sd(training_set[[col_name]]))
}

# Normalize training and test data
training_set <- normalize_data(training_set, columns_to_normalize, scaling_parameters)
test_set <- normalize_data(test_set, columns_to_normalize, scaling_parameters)

# Manually input values for prediction
new_data <- data.frame(Humidity = 26, Temperature = 95, Step.count = 168)
# Normalize the new data using the scaling parameters from the training set
new_data <- normalize_data(new_data, columns_to_normalize, scaling_parameters)

# Naive Bayes----
nb_model <- naiveBayes(Stress.Level ~ ., data = training_set)
# Ensure that nb_prediction has the same length as actual_values
nb_prediction <- predict(nb_model, newdata = new_data, type = "class")

# SVM----
svm_model <- svm(Stress.Level ~ ., data = training_set)
# Ensure that svm_prediction has the same length as actual_values
svm_prediction <- predict(svm_model, newdata = new_data)

# K-Nearest Neighbors Prediction----
knn_model <- knn(train = training_set[, columns_to_normalize], 
                 test = new_data, 
                 cl = training_set$Stress.Level, k = 3)
# Ensure that knn_model has the same length as actual_values
knn_prediction <- as.factor(knn_model)

# Multinomial Logistic Regression----
multinom_model <- multinom(Stress.Level ~ ., data = training_set, trace = FALSE)
# Ensure that multinom_prediction has the same length as actual_values
multinom_prediction <- predict(multinom_model, newdata = new_data, "class")

# Display Predictions
cat("Naive Bayes Prediction:", as.character(nb_prediction), "\n")
cat("Support Vector Machine Prediction:", as.character(svm_prediction), "\n")
cat("K-Nearest Neighbors Prediction:", as.character(knn_prediction), "\n")
cat("Multinomial Logistic Regression Prediction:", as.character(multinom_prediction), "\n")
