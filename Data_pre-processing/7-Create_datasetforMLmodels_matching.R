# =============================================================================
# 1. LOAD LIBRARIES & SET WORKING DIRECTORY
# =============================================================================
library(Matching)
library(dplyr)
library(MASS)

setwd("~/Summerproject/Descfriptive_analysis")

# =============================================================================
# 2. LOAD DATASETS
# =============================================================================
# Load imputation objects 
impu    <- load(file = "imputation.Rda")           
impucov <- load(file = "imputationcovariates.Rda")  
noimpu  <- readRDS("noimputationnew.rds")

data    <- as.data.frame(imputednew[[1]])
covdata <- as.data.frame(imputedcov[[1]])

datap <- merge(covdata, data, by = "eid")
datap <- merge(datap, noimpu, by = "eid")

# =============================================================================
# 3. SUBSET DATA: CLONE AND SYMBOL SELECTION
# =============================================================================
# Exclude records with clone equal to "small"
datap <- subset(datap, clone != "small")
datap$clone <- droplevels(datap$clone)

# Subset to include only records with symbol "No_mutation" or "JAK2"
datap <- subset(datap, symbol %in% c("No_mutation", "JAK2"))
datap$symbol <- droplevels(datap$symbol)

# Recode symbol so that "JAK2" becomes "TRUE" and "No_mutation" becomes "FALSE"
datap$symbol <- factor(datap$symbol, 
                       levels = c("JAK2", "No_mutation"), 
                       labels = c("TRUE", "FALSE"))

# Convert to numeric (1 for TRUE, 0 for FALSE) and then to logical
datap$symbol <- ifelse(datap$symbol == "TRUE", 1, 0)
levels(datap$symbol) <- c(TRUE, FALSE)
datap$symbol <- as.logical(datap$symbol)


# =============================================================================
# 4. MATCHING PROCEDURE
# =============================================================================
Tr <- datap$symbol
eid <- datap$eid

# Select matching covariates (sex, age, batch) and create a matching matrix
covariates <- datap %>% select(sex, age, batch)
X <- data.matrix(covariates)

# Perform 1:1 matching using the Matching package
rr <- Match(Y = eid, Tr = Tr, X = X, M = 1, ties = FALSE)

# Extract the matching results from rr[[6]]
new_list <- rr[[6]]

# Extract EID and matching numbers, then combine into one data frame
EID <- as.data.frame(new_list[[1]])
NUM <- as.data.frame(new_list[[3]])
together <- cbind(NUM, EID)

saveRDS(together, file = "together.rds")

# =============================================================================
# 5. MERGE MATCHING RESULTS WITH ORIGINAL DATA
# =============================================================================
blended <- together
Merged <- merge(blended, datap, by.x = "EID", by.y = "eid")
saveRDS(Merged, file = "geneJAK2OCT.rds")

# =============================================================================
# 6. CLEANING & FINAL DATA PREPARATION
# =============================================================================
setwd("~/Summerproject/Descfriptive_analysis")
merged_data <- readRDS("geneJAK2OCT.rds")

# Remove unwanted columns: EID, Had_menopause, sex.x, age.x, batch.x
Mergednew <- merged_data %>% 
  select(-EID, -Had_menopause, -sex.x, -age.x, -batch.x)

# Clean column names
colnames(Mergednew) <- gsub(pattern = "y*", replacement = "", x = colnames(Mergednew))
colnames(Mergednew) <- gsub(pattern = "\\.*", replacement = "", x = colnames(Mergednew))

datasetformodels <- Mergednew

setwd("~/Summerproject/outputs")

# Rename columns 
datasetformodels <- datasetformodels %>% 
  rename(
    Symbol = smbol,
    Sex    = sex,
    Age    = age,
    Batch  = batch,
    Pack_years_smoking = Pack_ears_smoking,
    Alcohol_intake_frequency = Alcohol_intake_frequenc
  )

datasetformodels <- datasetformodels %>% select(-CH, -clone)

datasetformodels$Alcohol_intake_frequency <- as.factor(datasetformodels$Alcohol_intake_frequency)

datasetformodels$Symbol <- factor(datasetformodels$Symbol, labels = c("No", "Yes"))

datasetformodels$Batch <- as.numeric(datasetformodels$Batch)
datasetformodels$Age   <- as.numeric(datasetformodels$Age)

# =============================================================================
# 7. COMBINE DATA FOR MODELING & SAVE FINAL DATASET
# =============================================================================
xdata <- datasetformodels %>% select(-BMI, -Symbol)
xall  <- datasetformodels %>% select(BMI, Symbol)
data_final <- cbind(xdata, xall)

saveRDS(data_final, file = "geneJAK2OCT.rds")

print(sum(is.na(data_final)))
write.csv(data_final, "~/Summerproject/outputs/python_geneJAK2oct.csv", row.names = FALSE)

