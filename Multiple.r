# Load libraries
library(ggplot2)
library(dplyr)
library(caTools)

# Load dataset
df <- read.csv("Student_Performance.csv", stringsAsFactors = TRUE)

# Clean column names
colnames(df) <- trimws(colnames(df))

# Convert categorical to numeric
df$ExtracurricularActivities <- ifelse(df$ExtracurricularActivities == "Yes", 1, 0)

# ---------------------------
# Train-Test Split (80-20)
# ---------------------------
set.seed(42)  # for reproducibility
split <- sample.split(df$PerformanceIndex, SplitRatio = 0.8)
train <- subset(df, split == TRUE)
test <- subset(df, split == FALSE)

# ---------------------------
# Train the model
# ---------------------------
model <- lm(PerformanceIndex ~ ., data = train)

# ---------------------------
# Predict on test data
# ---------------------------
test$Predicted <- predict(model, test)

# ---------------------------
# Evaluation: MSE and R-squared
# ---------------------------
mse <- mean((test$PerformanceIndex - test$Predicted)^2)
r_squared <- summary(model)$r.squared

cat("Mean Squared Error (MSE):", mse, "\n")
cat("R-squared on Training Set:", r_squared, "\n")

# ---------------------------
# 1. Actual vs Predicted Plot
# ---------------------------
ggplot(test, aes(x = PerformanceIndex, y = Predicted)) +
  geom_point(color = "blue") +
  geom_abline(slope = 1, intercept = 0, color = "red", linetype = "dashed") +
  labs(title = "Actual vs Predicted Performance Index (Test Data)",
       x = "Actual Performance Index",
       y = "Predicted Performance Index") +
  theme_minimal()

# ---------------------------
# 2. Feature Coefficient Bar Plot
# ---------------------------
coeffs <- data.frame(
  Feature = names(model$coefficients)[-1],
  Coefficient = model$coefficients[-1]
)

ggplot(coeffs, aes(x = reorder(Feature, abs(Coefficient)), y = Coefficient)) +
  geom_bar(stat = "identity", fill = "skyblue", color = "black") +
  coord_flip() +
  labs(title = "Feature Importance (Coefficients)",
       x = "Feature",
       y = "Coefficient Value") +
  theme_minimal()