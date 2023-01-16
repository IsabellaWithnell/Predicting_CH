library(ggplot2)
library(dplyr)
library(tidyr)

#I created 1 and 2 below as separate scripts and submitted them separately to the HPC

#1: Impute blood data and BMI

library(ggplot2)
library(dplyr)
library(tidyr)

setwd("~/Summerproject/Descfriptive_analysis")
datap <- readRDS("January_blood_dataset.rds")
colnames(datap)
datap <- datap %>% dplyr::select(eid, WBC,RBC,HT, MCV, HGB, RDW, PLT, PCT, PDW, LY, MO, NE, EO, RET , HLR , BMI)
colnames(datap)

datap$eid <- as.numeric(datap$eid)

datap <- data.matrix(datap)

library(impute)

imputednew <- impute.knn(datap, k=10 , rowmax = 0.5, colmax = 0.8,maxp = 1500, rng.seed=362436069)

save(imputednew, file="imputation.Rda")

#2: No imputation variables


library(ggplot2)
library(dplyr)
library(tidyr)

setwd("~/Summerproject/Descfriptive_analysis")
datap <- readRDS("January_blood_dataset.rds")
colnames(datap)
datap <- datap %>% dplyr::select(eid, Alcohol_intake_frequency., Alcohol_drink_status, Current_tobacco_smoking,
                                 Time_since_last_menstrual_period, Had_menopause, Smoking_status, Pack_years_smoking,
                                 Batch_b001, Sex, age, symbol, largeclone01, largeclone015, largeclone02, -clone)

colnames(datap)

saveRDS(datap, file="No_impute.rds")




