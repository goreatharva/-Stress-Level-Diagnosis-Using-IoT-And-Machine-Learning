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
cat("Checking the null values:", p)
cat("\nThere are no null values in our dataset.\n")

# Encoding the target feature as factor
dataset$Stress.Level <- factor(dataset$Stress.Level, levels = c(0, 1, 2))

# Data normalization
column_to_normalize <- c("Humidity", "Temperature", "Step.count")

for (col_name in column_to_normalize) {
  column_to_normalize <- dataset[, col_name]
  normalized_column <- scale(column_to_normalize)
  range <- (normalized_column - min(normalized_column)) / (max(normalized_column) - min(normalized_column))
  dataset[, col_name] <- range
}

# Divide into train and test data
set.seed(123)
split <- sample.split(dataset$Stress.Level, SplitRatio = 0.75)
training_set <- subset(dataset, split == TRUE)
test_set <- subset(dataset, split == FALSE)

# Naive Bayes
nb_model <- naiveBayes(Stress.Level ~ ., data = training_set)
nb_predictions <- predict(nb_model, newdata = test_set, type = "class")
cm_nb <- confusionMatrix(nb_predictions, test_set$Stress.Level)

# Print Confusion Matrix for Naive Bayes
cat("\nConfusion Matrix (Naive Bayes):\n")
print(table(Reference = as.factor(test_set$Stress.Level), Prediction = as.factor(nb_predictions)))

# Print Naive Bayes Metrics
cat("Naive Bayes Metrics:\n")
cat("Accuracy:", round(cm_nb$overall["Accuracy"] * 100, 1), "%\n")
cat("Precision:", cm_nb$byClass[2, "Pos Pred Value"], "\n")
cat("Recall:", cm_nb$byClass[2, "Sensitivity"], "\n")
cat("F1-Score:", cm_nb$byClass[2, "F1"], "\n")

# CAP for Naive Bayes
nb_probs <- predict(nb_model, newdata = test_set[-4], type = "raw")
nb_probs_class1 <- nb_probs[, 2]
nb_sorted_data_nb <- data.frame(probs = nb_probs_class1, actual = test_set$Stress.Level)
nb_sorted_data_nb <- nb_sorted_data_nb[order(-nb_sorted_data_nb$probs), ]
nb_sorted_data_nb$index <- 1:nrow(nb_sorted_data_nb)
nb_sorted_data_nb$actual <- as.numeric(levels(nb_sorted_data_nb$actual))[nb_sorted_data_nb$actual]
nb_cumulative_actual_nb <- cumsum(nb_sorted_data_nb$actual)
nb_cap_curve <- data.frame(
  x = c(0, nb_sorted_data_nb$index, nrow(test_set)),
  y = c(0, nb_cumulative_actual_nb, sum(test_set$Stress.Level == 1))
)

# Print CAP Curve for Naive Bayes
plot_nb <- ggplot() +
  geom_line(data = nb_cap_curve, aes(x, y, color = "Naive Bayes Classifier", linetype = "Naive Bayes Classifier")) +
  labs(title = "Cumulative Accuracy Profile (Naive Bayes)", x = "Total observations", y = "Class 1 observations") +
  theme_minimal() +
  theme(legend.position = "right") +
  scale_color_manual(values = c("Random Model" = "red", "Perfect Model" = "grey", "Naive Bayes Classifier" = "green"), labels = c("Random Model", "Perfect Model", "Naive Bayes Classifier"))

print(plot_nb)

# Logistic Regression
model_lr <- multinom(Stress.Level ~ ., data = training_set)
y_pred_lr <- predict(model_lr, newdata = test_set, type = "class")
cm_lr <- confusionMatrix(y_pred_lr, test_set$Stress.Level)$table
cat("\nConfusion Matrix (Logistic Regression):\n")
print(cm_lr)

# Extracting relevant metrics
precision_lr <- ifelse(sum(cm_lr[, 1]) == 0, 0, cm_lr[1, 1] / sum(cm_lr[, 1]))
recall_lr <- ifelse(sum(cm_lr[1, ]) == 0, 0, cm_lr[1, 1] / sum(cm_lr[1, ]))
f1_score_lr <- ifelse((precision_lr + recall_lr) == 0, 0, 2 * (precision_lr * recall_lr) / (precision_lr + recall_lr))
accuracy_lr <- sum(diag(cm_lr)) / sum(cm_lr)

cat("Logistic Regression Metrics:\n")
cat("Accuracy:", round(accuracy_lr * 100, 2), "%\n")
cat("Precision:", precision_lr, "\n")
cat("Recall:", recall_lr, "\n")
cat("F1-Score:", f1_score_lr, "\n")


# CAP for Logistic Regression
probs_lr <- predict(model_lr, newdata = test_set, type = "probs")
probs_lr <- probs_lr[, "1"]  # Probability of class 1
sorted_data_lr <- data.frame(probs = probs_lr, actual = test_set$Stress.Level)
sorted_data_lr <- sorted_data_lr[order(-sorted_data_lr$probs), ]
sorted_data_lr$index <- 1:nrow(sorted_data_lr)
sorted_data_lr$actual <- as.numeric(levels(sorted_data_lr$actual))[sorted_data_lr$actual]
cumulative_actual_lr <- cumsum(sorted_data_lr$actual)
cap_curve_lr <- data.frame(
  x = c(0, sorted_data_lr$index, nrow(test_set)),
  y = c(0, cumulative_actual_lr, sum(test_set$Stress.Level == 1))
)

# Print CAP Curve for Logistic Regression
plot_lr <- ggplot() +
  geom_line(data = cap_curve_lr, aes(x, y, color = "Logistic Regression Classifier", linetype = "Logistic Regression Classifier")) +
  labs(title = "Cumulative Accuracy Profile (Logistic Regression)", x = "Total observations", y = "Class 1 observations") +
  theme_minimal() +
  theme(legend.position = "right")

print(plot_lr)

# KNN
k <- 5
knn_model <- knn(train = training_set[, -4], test = test_set[, -4], cl = training_set$Stress.Level, k = k)
cm_knn <- table(Actual = test_set$Stress.Level, Predicted = knn_model)
cat("\nConfusion Matrix (KNN):\n")
print(cm_knn)

# KNN
# KNN
precision_knn <- ifelse(sum(cm_knn[, 2]) == 0, 0, cm_knn[2, 2] / sum(cm_knn[, 2]))
recall_knn <- ifelse(sum(cm_knn[2, ]) == 0, 0, cm_knn[2, 2] / sum(cm_knn[2, ]))
f1_score_knn <- ifelse((precision_knn + recall_knn) == 0, 0, 2 * (precision_knn * recall_knn) / (precision_knn + recall_knn))
accuracy_knn <- sum(diag(cm_knn)) / sum(cm_knn)

cat("KNN Metrics:\n")
cat("Accuracy:", round(accuracy_knn * 100, 2), "%\n")
cat("Precision:", precision_knn, "\n")
cat("Recall:", recall_knn, "\n")
cat("F1-Score:", f1_score_knn, "\n")

# CAP for KNN
knn_model <- knn(train = training_set[, -4], test = test_set[, -4], cl = training_set$Stress.Level, k = k)
knn_correct <- knn_model == test_set$Stress.Level
knn_cumulative_correct <- cumsum(knn_correct)
cap_curve_knn <- data.frame(
  x = c(0, 1:length(knn_cumulative_correct), nrow(test_set)),
  y = c(0, knn_cumulative_correct, sum(test_set$Stress.Level == 1))
)

# Print CAP Curve for KNN
plot_knn <- ggplot() +
  geom_line(data = cap_curve_knn, aes(x, y, color = "KNN Classifier", linetype = "KNN Classifier")) +
  labs(title = "Cumulative Accuracy Profile (KNN)", x = "Total observations", y = "Class 1 observations") +
  theme_minimal() +
  theme(legend.position = "right")

print(plot_knn)

# SVM
svm_model <- svm(Stress.Level ~ ., data = training_set, kernel = "linear", verbose = FALSE)
svm_predictions <- predict(svm_model, newdata = test_set)
cm_svm <- confusionMatrix(svm_predictions, test_set$Stress.Level)

# Print Confusion Matrix for SVM
cat("\nConfusion Matrix (SVM):\n")
print(table(Reference = as.factor(test_set$Stress.Level), Prediction = as.factor(svm_predictions)))

# Calculate and Print Metrics for SVM
precision_svm <- ifelse(sum(cm_svm$byClass[, "Pos Pred Value"]) == 0, 0, cm_svm$byClass[2, "Pos Pred Value"])
recall_svm <- ifelse(sum(cm_svm$byClass[, "Sensitivity"]) == 0, 0, cm_svm$byClass[2, "Sensitivity"])
f1_score_svm <- ifelse((precision_svm + recall_svm) == 0, 0, 2 * (precision_svm * recall_svm) / (precision_svm + recall_svm))
accuracy_svm <- cm_svm$overall["Accuracy"]

cat("SVM Metrics:\n")
cat("Accuracy:", round(accuracy_svm * 100, 2), "%\n")
cat("Precision:", precision_svm, "\n")
cat("Recall:", recall_svm, "\n")
cat("F1-Score:", f1_score_svm, "\n")

# CAP for SVM
svm_probs <- predict(svm_model, newdata = test_set, decision.values = TRUE)
svm_sorted_data <- data.frame(probs = as.numeric(levels(svm_probs))[svm_probs], actual = test_set$Stress.Level)
svm_sorted_data <- svm_sorted_data[order(-svm_sorted_data$probs), ]
svm_sorted_data$index <- 1:nrow(svm_sorted_data)
svm_sorted_data$actual <- as.numeric(levels(svm_sorted_data$actual))[svm_sorted_data$actual]
svm_cumulative_actual <- cumsum(svm_sorted_data$actual)

svm_cap_curve <- data.frame(
  x = c(0, svm_sorted_data$index, nrow(test_set)),
  y = c(0, svm_cumulative_actual, sum(test_set$Stress.Level == 1))
)

# Print CAP Curve for SVM
plot_svm <- ggplot() +
  geom_line(data = svm_cap_curve, aes(x, y, color = "SVM Classifier", linetype = "SVM Classifier")) +
  labs(title = "Cumulative Accuracy Profile (SVM)", x = "Total observations", y = "Class 1 observations") +
  theme_minimal() +
  theme(legend.position = "right")

print(plot_svm)
