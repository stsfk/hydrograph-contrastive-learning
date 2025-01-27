library(tidyverse)

# Read from raw data ------------------------------------------------------
library(fs)

main_folder <- "/Volumes/Data/data/raw/Caravan/1.4/timeseries/csv/"

files <- dir_ls(main_folder, recurse = TRUE, type = "file")

read_and_select <- function(file_path) {
  read_csv(file_path, show_col_types = FALSE) %>%
    select(
      Date = date,
      P = total_precipitation_sum,
      PET = potential_evaporation_sum,
      T = temperature_2m_mean,
      Q = streamflow
    ) %>%
    suppressWarnings()
}


# Create an empty list to store the data
data_list <- list()

# Loop over each file
for (file in files) {
  tryCatch({
    # Read the file using fread from data.table
    data <- read_and_select(file)
    
    # Store the data in the list with the file name as the list name
    file_name <- basename(file)
    data_list[[file_name]] <- data
  }, error = function(e) {
    # Handle errors, such as when the data exceeds memory capacity
    warning(paste("Error reading file:", file))
  })
}

# save data
save(data_list, file = "./data/forcing&Q_raw.Rda")



# Save raw Q data ---------------------------------------------------------

load("./data/forcing&Q_raw.Rda")

data_process <- tibble(
  catchment_name =  names(data_list) %>% str_sub(1, -5),
  data = lapply(data_list, function(x)
    x %>% dplyr::select(Date, Q))
)

save(data_process, file = "./data/Q_raw.Rda")


# Filter catchments -----------------------------------------------------
library(zoo)
library(furrr)

load("./data/Q_raw.Rda")

# Define the periods
training_period <- c(as.Date('1980-01-01'), as.Date('1995-12-31'))
validation_period <- c(as.Date('1996-01-01'), as.Date('2005-12-31'))
test_period <- c(as.Date('2006-01-01'), as.Date('2023-12-31'))

# Filter the data to create subsets for processing
plan(multisession, workers = 6)

data_processing_training <- data_process %>%
  mutate(data = future_map(data, function(x)
    x %>% filter(Date >= training_period[1], Date <= training_period[2])))

data_processing_validation <- data_process %>%
  mutate(data = future_map(data, function(x)
    x %>% filter(Date >= validation_period[1], Date <= validation_period[2])))

data_processing_test <- data_process %>%
  mutate(data = future_map(data, function(x)
    x %>% filter(Date >= test_period[1], Date <= test_period[2])))

rm(data_process) # for memory saving

future::plan(future::sequential)


# Count the number of non-missings
plan(multisession, workers = 8)
data_processing_training <- data_processing_training %>%
  mutate(
    data = future_map(data, function(x)
      x %>% mutate(Q_non_missing = !is.na(Q))),
    data = future_map(data, function(x)
      x %>% mutate(
        Q_missing_roll_sum = rollapply(Q_non_missing, 365, sum, partial = TRUE, align = "left")
      )),
    data = future_map(data, function(x)
      x %>% mutate(eligibility = (
        Q_missing_roll_sum == 365
      )))
  )
future::plan(future::sequential)


plan(multisession, workers = 8)
data_processing_validation <- data_processing_validation %>%
  mutate(
    data = future_map(data, function(x)
      x %>% mutate(Q_non_missing = !is.na(Q))),
    data = future_map(data, function(x)
      x %>% mutate(
        Q_missing_roll_sum = rollapply(Q_non_missing, 365, sum, partial = TRUE, align = "left")
      )),
    data = future_map(data, function(x)
      x %>% mutate(eligibility = (
        Q_missing_roll_sum == 365
      )))
  )
future::plan(future::sequential)


plan(multisession, workers = 8)
data_processing_test <- data_processing_test %>%
  mutate(
    data = future_map(data, function(x)
      x %>% mutate(Q_non_missing = !is.na(Q))),
    data = future_map(data, function(x)
      x %>% mutate(
        Q_missing_roll_sum = rollapply(Q_non_missing, 365, sum, partial = TRUE, align = "left")
      )),
    data = future_map(data, function(x)
      x %>% mutate(eligibility = (
        Q_missing_roll_sum == 365
      )))
  )
future::plan(future::sequential)


# Filter catchments based on eligible days threshold
data_processing_training <- data_processing_training %>%
  mutate(eligible_days = map_dbl(data, function(x)
    x$eligibility %>% sum()))
data_processing_validation <- data_processing_validation %>%
  mutate(eligible_days = map_dbl(data, function(x)
    x$eligibility %>% sum()))
data_processing_test <- data_processing_test %>%
  mutate(eligible_days = map_dbl(data, function(x)
    x$eligibility %>% sum()))


threshold <- 365 * 5
catchments_retained <- tibble(
  catchment_name = data_processing_training$catchment_name,
  eligible_days_training = data_processing_training$eligible_days,
  eligible_days_validation = data_processing_validation$eligible_days,
  eligible_days_test = data_processing_test$eligible_days
) %>%
  filter(
    eligible_days_training >= threshold,
    eligible_days_validation >= threshold,
    eligible_days_test >= threshold
  )

# Number of catchments retained: 5,623
data_processing_training <- catchments_retained %>%
  select(catchment_name) %>%
  left_join(data_processing_training %>% select(catchment_name, data), by = "catchment_name")

data_processing_validation <- catchments_retained %>%
  select(catchment_name) %>%
  left_join(data_processing_validation %>% select(catchment_name, data),
            by = "catchment_name")

data_processing_test <- catchments_retained %>%
  select(catchment_name) %>%
  left_join(data_processing_test %>% select(catchment_name, data), by = "catchment_name")

# save data
save(data_processing_training,
     data_processing_validation,
     data_processing_test,
     file = "./data/Q_filtered.Rda")


# Prepare modeling data ---------------------------------------------------

load("./data/Q_filtered.Rda")

data_processing_training <- data_processing_training %>%
  unnest(data) %>%
  select(Date, Q, catchment_name, eligibility)

data_processing_validation <- data_processing_validation %>%
  unnest(data) %>%
  select(Date, Q, catchment_name, eligibility)

data_processing_test <- data_processing_test %>%
  unnest(data) %>%
  select(Date, Q, catchment_name, eligibility)

write_csv(data_processing_training, file = 'data/training_data.csv')
write_csv(data_processing_validation, file = 'data/validation_data.csv')
write_csv(data_processing_test, file = 'data/test_data.csv')


# Analyze data ------------------------------------------------------------
library(zoo)

training_tibble <- read_csv('data/training_data.csv')
validation_tibble <- read_csv("data/validation_data.csv")
test_tibble <- read_csv("data/test_data.csv")

# check if the eligible index are indeed eligible, i.e., no missing values when using it as the start index to create hydrograph

training_tibble <- training_tibble %>%
  group_by(catchment_name) %>%
  mutate(Q_roll_sum = rollapply(Q, 365, sum, partial = TRUE, align = "left")) %>%
  ungroup()

validation_tibble <- validation_tibble %>%
  group_by(catchment_name) %>%
  mutate(Q_roll_sum = rollapply(Q, 365, sum, partial = TRUE, align = "left")) %>%
  ungroup()

test_tibble <- test_tibble %>%
  group_by(catchment_name) %>%
  mutate(Q_roll_sum = rollapply(Q, 365, sum, partial = TRUE, align = "left")) %>%
  ungroup()

training_tibble %>% filter(eligibility) %>% pull(Q_roll_sum) %>% sum()
validation_tibble %>% filter(eligibility) %>% pull(Q_roll_sum) %>% sum()
test_tibble %>% filter(eligibility) %>% pull(Q_roll_sum) %>% sum()

# count the total number of eligibility
training_tibble$eligibility %>% sum() # 27,336,327 possible 365-day hydrographs
validation_tibble$eligibility %>% sum() # 18,093,723 possible 365-day hydrographs
test_tibble$eligibility %>% sum() # 2,097,0184 possible 365-day hydrographs



