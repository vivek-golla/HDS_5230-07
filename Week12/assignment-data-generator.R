library(mlbench)
library(purrr)
library(data.table)

data("PimaIndiansDiabetes2")
ds <- as.data.frame(na.omit(PimaIndiansDiabetes2))

# Fit logistic regression to derive coefficients
logmodel <- glm(diabetes ~ ., data = ds, family = "binomial")
cfs <- coefficients(logmodel)
prednames <- variable.names(ds)[-9]  # predictors only

# Function to generate synthetic data
generate_dataset <- function(sz, file_name) {
  dfdata <- map_dfc(prednames,
                    function(nm) {
                      sample(ds[[nm]], size = sz, replace = TRUE)
                    })
  names(dfdata) <- prednames
  
  # Compute logit
  pvec <- map(1:8, function(pnum) {
    cfs[pnum + 1] * dfdata[[prednames[pnum]]]
  }) %>% reduce(`+`) + cfs[1]
  
  # Assign binary outcome
  dfdata$outcome <- ifelse(1 / (1 + exp(-pvec)) > 0.5, 1, 0)
  
  # Write to CSV
  fwrite(dfdata, file_name)
}

# Create datasets
generate_dataset(1000, "synthetic_1000.csv")
generate_dataset(10000, "synthetic_10000.csv")
generate_dataset(100000, "synthetic_100000.csv")