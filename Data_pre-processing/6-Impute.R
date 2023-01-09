library(ggplot2)
library(dplyr)
library(tidyr)

#I created all of 1,2 and 3 below as separate scripts and submitted them separately to the HPC

#1: Impute blood data

setwd("~/Summerproject/Descfriptive_analysis")
datap <- readRDS("Blood_finalised_new_together_oct_edit_2.rds")
colnames(datap)
datap <- datap %>% dplyr::select(eid, WBC,RBC,HT, MCV, HGB, RDW, PLT, PCT, PDW, LY, MO, NE, EO, RET , HLR )
colnames(datap)

datap$eid <- as.numeric(datap$eid)

datap <- data.matrix(datap)

library(impute)

imputednew <- impute.knn(datap, k=10 , rowmax = 0.5, colmax = 0.8,maxp = 1500, rng.seed=362436069)

save(imputednew, file="imputation.Rda")

#2:Impute covariates

setwd("~/Summerproject/Descfriptive_analysis")
datap <- readRDS("Blood_finalised_new_together_oct_edit_2.rds")
colnames(datap)
datap <- datap %>% dplyr::select(-WBC, -RBC, -HT, -MCV, -HGB, -RDW, -PLT, 
                                 -PCT, -PDW, -LY, -MO,  -NE, -EO, -RET, -HLR, -Alcohol_intake_frequency., -Alcohol_drink_status,
                                 -Current_tobacco_smoking, -Time_since_last_menstrual_period, -Had_menopause, -Smoking_status, -Pack_years_smoking,
                                  -batch, -sex, -age, -symbol, -clone, -CH)
colnames(datap)
sum(is.na(datap$Pack_years_smoking))
datap$eid <- as.numeric(datap$eid)
datap <- data.matrix(datap)

library(impute)

imputedcov <- impute.knn(datap, k=10 , rowmax = 0.5, colmax = 0.8,maxp = 1500, rng.seed=362436069)

save(imputedcov, file="imputationcovariates.Rda")

#3: No imputation variables


setwd("~/Summerproject/Descfriptive_analysis")
datap <- readRDS("Blood_finalised_new_together_oct_edit_2.rds")
colnames(datap)
datap <- datap %>% dplyr::select(eid, Alcohol_intake_frequency., Alcohol_drink_status, Current_tobacco_smoking,
                                 Time_since_last_menstrual_period, Had_menopause, Smoking_status, Pack_years_smoking,
                                 batch, sex, age, symbol, clone, CH)

colnames(datap)

saveRDS(datap, file="noimputationnew.rds")



