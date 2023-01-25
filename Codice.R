## ZULATO DAVIDE mat. 876101

# XGBoost su dataset Home sales prices, previsione del prezzo di vendita 
# (price, in scala log10) di 4320 abitazioni del test set. 

rm(list=ls())

# ==============================================================================
# 1. caricamento librerie e dataset

library(readr)
library(tidymodels)
library(tidyverse)
library(tidymodels)
library(modeldata)
library(doParallel)

PATH <- "https://raw.githubusercontent.com/aldosolari/DM/master/docs/HomePrices/"
train = read_csv2(paste0(PATH,"home_prices_train.csv"))
test = read_csv2(paste0(PATH,"home_prices_test.csv"))

# parallel processing
all_cores <- parallel::detectCores(logical = FALSE) 
registerDoParallel(cores = all_cores)

# ==============================================================================
# 2. Preprocessing, ricetta

preprocessing_recipe <- 
  recipe(price ~ ., data = train) %>%
  # combine low frequency factor levels (condition)
  step_other(condition, threshold = 0.01) %>% 
  # convert character and factor into dummy
  step_dummy(all_nominal_predictors()) %>%
  # remove no variance predictors which provide no predictive information
  recipes::step_nzv(all_nominal()) %>%
  # Center and scale for PCA
  step_center(all_numeric_predictors()) %>%
  step_scale(all_numeric_predictors()) %>%
  # PCA (improves XGBoost performances)
  step_pca(sqft_above, sqft_basement,sqft_living,sqft_lot,nn_sqft_living,nn_sqft_lot,year_renovated,yr_built)%>%
  prep()


# ==============================================================================
# 3. XGBoost Model Specification. MODELLO

xgboost_model <- 
  parsnip::boost_tree(
    mode = "regression",
    trees = 1000,
    min_n = 38, #numero minimo di obs per effettuare split
    tree_depth = 13,
    learn_rate = 0.0199035673943381,
    loss_reduction = 2.43502192814882e-10
  ) %>%
  set_engine("xgboost", objective = "reg:squarederror") #funzione obiettivo 

# ==============================================================================
# 4. Previsione su Test Data

train_processed <- bake(preprocessing_recipe,  new_data = train)
test_processed  <- bake(preprocessing_recipe, new_data = test)

test_prediction <- xgboost_model %>%
  # fit del modello sui dati di training
  fit(
    formula = price ~ ., 
    data    = train_processed
  ) %>%
  # fit del modello per previsione su test pre-processato
  predict(new_data = test_processed) %>%
  bind_cols(test)


# previsioni su file txt
write.table(file="876101_previsione.txt", test_prediction$.pred, row.names = FALSE, col.names = FALSE)


