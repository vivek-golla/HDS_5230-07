Week 11 Assignment Final R
================
2025-04-17

``` r
library(xgboost)
```

    ## Warning: package 'xgboost' was built under R version 4.4.3

``` r
library(mlbench)
```

    ## Warning: package 'mlbench' was built under R version 4.4.2

``` r
library(data.table)
library(caret)
```

    ## Warning: package 'caret' was built under R version 4.4.3

    ## Loading required package: ggplot2

    ## Warning: package 'ggplot2' was built under R version 4.4.2

    ## Loading required package: lattice

``` r
# library(microbenchmark)
library(purrr)
```

    ## 
    ## Attaching package: 'purrr'

    ## The following object is masked from 'package:caret':
    ## 
    ##     lift

    ## The following object is masked from 'package:data.table':
    ## 
    ##     transpose

``` r
dfdata <- read.csv("dfdata.csv")
```

``` r
dt <- data.table(dfdata)
dt <- na.omit(dt)

set.seed(2025)
tr.rows <- createDataPartition(y = dt$outcome,
                               p = 0.8)
```

``` r
# Define sample fractions
frac_values <- c(0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1)

# Initialize results list
results <- data.frame(
  Frac = numeric(),
  SampleSize = integer(),
  Accuracy = numeric(),
  Time = numeric(),
  stringsAsFactors = FALSE
)
```

# First approach - basic xgboost with basic Cross Validation

``` r
# Loop through fractions
for (frac in frac_values) {
  cat("\nRunning frac =", frac, "...\n")
  
  # Subsample data
  dt_sub <- dt[sample(.N, size = ceiling(.N * frac), replace = FALSE)]
  
  # Ensure both classes are represented
  if (length(unique(dt_sub$outcome)) < 2) {
    cat("Skipping frac =", frac, "- not enough class variability.\n")
    next
  }
  
  # Train/Test Split (80/20)
  set.seed(2025)
  tr_idx <- createDataPartition(dt_sub$outcome, p = 0.8, list = FALSE)
  train <- dt_sub[tr_idx]
  test  <- dt_sub[-tr_idx]

  # Prepare matrices for xgboost
  x_train <- as.matrix(train[, !"outcome"])
  y_train <- train$outcome
  x_test  <- as.matrix(test[, !"outcome"])
  y_test  <- test$outcome

  # Timer start
  t0 <- Sys.time()

  # Fit XGBoost model
  model <- xgboost(data = x_train,
                   label = y_train,
                   objective = "binary:logistic",
                   nrounds = 50,
                   verbose = 0)

  # Predict
  preds <- predict(model, x_test)
  y_pred <- ifelse(preds > 0.5, 1, 0)

  # Timer end
  t1 <- Sys.time()
  
  acc <- mean(y_pred == y_test)
  duration <- round(as.numeric(difftime(t1, t0, units = "secs")), 2)

  # Append result
  results <- rbind(results, data.frame(
    Frac = frac,
    SampleSize = nrow(dt_sub),
    Accuracy = round(acc, 4),
    Time = duration
  ))
}
```

    ## 
    ## Running frac = 1e-06 ...
    ## 
    ## Running frac = 1e-05 ...
    ## 
    ## Running frac = 1e-04 ...
    ## 
    ## Running frac = 0.001 ...
    ## 
    ## Running frac = 0.01 ...
    ## 
    ## Running frac = 0.1 ...

``` r
# Print results table
print(results)
```

    ##    Frac SampleSize Accuracy   Time
    ## 1 1e-06        100   0.8500   0.32
    ## 2 1e-05       1001   0.9600   0.35
    ## 3 1e-04      10000   0.9720   0.62
    ## 4 1e-03     100000   0.9842   1.63
    ## 5 1e-02    1000000   0.9885  19.02
    ## 6 1e-01   10000000   0.9896 110.27

# Second approach - xgboost using caret with 5-fold Cross Validation

``` r
# Define sample fractions
frac_values <- c(0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1)

# Initialize results list
results2 <- data.frame(
  Frac = numeric(),
  SampleSize = integer(),
  Accuracy = numeric(),
  Time = numeric(),
  stringsAsFactors = FALSE
)
```

``` r
for (frac in frac_values) {
  cat("\nRunning frac =", frac, "...\n")
  
  # Subsample data
  dt_sub <- dt[sample(.N, size = ceiling(.N * frac), replace = FALSE)]
  
  #explicitly specifying outcome as factor so i can use in caret
  dt_sub[, outcome := factor(outcome, levels = c(0, 1), labels = c("No", "Yes"))]
  
  # Ensure both classes are represented
  if (length(unique(dt_sub$outcome)) < 2) {
    cat("Skipping frac =", frac, "- not enough class variability.\n")
    next
  }
  set.seed(2025)
  trainIndices <- createDataPartition(dt_sub$outcome,
                                    p = 0.8)
  training <- dt_sub[trainIndices$Resample1, ]
  holdout <- dt_sub[-trainIndices$Resample1, ]

  fitControl <- trainControl(
  method = "repeatedcv", ## perform repeated k-fold CV
  number = 5,
  repeats = 1)

  # Timer start
  t0 <- Sys.time()
  
  xgbfit <- train(outcome ~ .,
                     data = training,
                     method = "xgbTree",
                     trControl = fitControl,
                    tuneGrid = expand.grid(
                  nrounds = 50,
                  max_depth = 6,
                  eta = 0.3,
                  gamma = 0,
                  colsample_bytree = 1,
                  min_child_weight = 1,
                  subsample = 1
                                          ),
                     verbose = FALSE,
                  
                     )

  predvals <- predict(xgbfit, holdout)
  
  predvals <- factor(predvals, levels = c("No", "Yes"))
  holdout_outcome <- factor(holdout$outcome, levels = c("No", "Yes"))
  
  ## create the confusion matrix and view the results
  cm <- confusionMatrix(predvals, holdout$outcome)
  acc <- cm$overall["Accuracy"]
  
  # Timer end
  t1 <- Sys.time()
  
  duration <- round(as.numeric(difftime(t1, t0, units = "secs")), 2)

  # Append result
  results2 <- rbind(results2, data.frame(
    Frac = frac,
    SampleSize = nrow(dt_sub),
    Accuracy = round(acc, 4),
    Time = duration
  ))
}
```

    ## 
    ## Running frac = 1e-06 ...
    ## 
    ## Running frac = 1e-05 ...
    ## 
    ## Running frac = 1e-04 ...
    ## 
    ## Running frac = 0.001 ...
    ## 
    ## Running frac = 0.01 ...
    ## 
    ## Running frac = 0.1 ...

``` r
# Print results table
print(results2)
```

    ##            Frac SampleSize Accuracy   Time
    ## Accuracy  1e-06        100   0.7368   1.96
    ## Accuracy1 1e-05       1001   0.9200   1.86
    ## Accuracy2 1e-04      10000   0.9750   2.97
    ## Accuracy3 1e-03     100000   0.9844   8.21
    ## Accuracy4 1e-02    1000000   0.9888 129.30
    ## Accuracy5 1e-01   10000000   0.9897 689.55
