df <- read.csv("Experience-Salary.csv")

# Model
model <- lm(Salary ~ YearsExperience, data=df)
summary(model)

# Plot
plot(df$YearsExperience, df$Salary, pch=19, col='blue', main='Experience vs Salary')
abline(model, col='red', lwd=2)
