rm(list=ls())
library(dplyr)
library(ggplot2)
library(tidyr)
library(forcats)
library(table1)
library(tidyverse)
conflict_prefer("rename", "dplyr")


#### Code to clean the data. Includes other variables that were used for exploratory analysis (deleted here)

setwd("/rds/general/user/iw413/home/Summerproject/outputs")
data <- readRDS("ukb_recoded.rds")
data_2 <- readRDS("ukb_extracted.rds")
ukb_final <- readRDS("ukb_final.rds")

datafinal <-data %>% mutate(eid=rownames(data_2))

setwd("/rds/general/user/iw413/home/Summerproject/extraction_and_recoding")
dff3<- datafinal
dff1<-readRDS("output_final_malig.rds")%>% 
  mutate(eid = as.character(eid)) 
malig_ukbb<-left_join(dff3,dff1,by="eid")  
colnames(malig_ukbb)

malig_ukbb$Time_since_last_menstrual_period.0.0
malig_ukbb$Alcohol_intake_yesterday <- malig_ukbb$Alcohol_intake_yesterday.0.0

malig_ukbb <- within(malig_ukbb, {   
  Time_since_last_menstrual_period <- NA # need to initialize variable
  Time_since_last_menstrual_period[Time_since_last_menstrual_period.0.0 ==-1 ] <- "Unknown"
  Time_since_last_menstrual_period[Time_since_last_menstrual_period.0.0 ==-3 ] <- "Unknown"
  Time_since_last_menstrual_period[Time_since_last_menstrual_period.0.0 <3  & Time_since_last_menstrual_period.0.0 >= 0] <- "Under_3days"
  Time_since_last_menstrual_period[Time_since_last_menstrual_period.0.0 <7 & Time_since_last_menstrual_period.0.0 >= 3] <- "3days_to_1week"
  Time_since_last_menstrual_period[Time_since_last_menstrual_period.0.0 <14 & Time_since_last_menstrual_period.0.0 >= 7] <- "1week_to_2weeks"
  Time_since_last_menstrual_period[Time_since_last_menstrual_period.0.0 <21 & Time_since_last_menstrual_period.0.0 >= 14] <- "2weeks_to_3weeks"
  Time_since_last_menstrual_period[Time_since_last_menstrual_period.0.0 >= 21] <- "Over_3weeks"
  Time_since_last_menstrual_period[is.na(Time_since_last_menstrual_period.0.0) & Sex.0.0 == "Female"] <- "Female_menopause"
  Time_since_last_menstrual_period[is.na(Time_since_last_menstrual_period.0.0) & Sex.0.0 == "Male"] <- "Male"
  
} )

malig_ukbb$Time_since_last_menstrual_period <- as.factor(malig_ukbb$Time_since_last_menstrual_period) 


malig_ukbb$Alcohol_consumed<- coalesce(malig_ukbb$Alcohol_consumed.0.0, malig_ukbb$Alcohol_consumed.1.0, 
                                       malig_ukbb$Alcohol_consumed.2.0, malig_ukbb$Alcohol_consumed.3.0, 
                                       malig_ukbb$Alcohol_consumed.4.0)

malig_ukbb$Alcohol_intake_yesterday<- coalesce(malig_ukbb$Alcohol_intake_yesterday.0.0, malig_ukbb$Alcohol_intake_yesterday.1.0, 
                                               malig_ukbb$Alcohol_intake_yesterday.2.0, malig_ukbb$Alcohol_intake_yesterday.3.0, 
                                               malig_ukbb$Alcohol_intake_yesterday.4.0)


malig_ukbb$Chemical_workplace<- coalesce(malig_ukbb$Chemical_workplace.0.0, malig_ukbb$Chemical_workplace.0.1, 
                                         malig_ukbb$Chemical_workplace.0.2, malig_ukbb$Chemical_workplace.0.3, 
                                         malig_ukbb$Chemical_workplace.0.4, malig_ukbb$Chemical_workplace.0.5, malig_ukbb$Chemical_workplace.0.6, 
                                         malig_ukbb$Chemical_workplace.0.7, malig_ukbb$Chemical_workplace.0.8, malig_ukbb$Chemical_workplace.0.9, 
                                         malig_ukbb$Chemical_workplace.0.10, malig_ukbb$Chemical_workplace.0.11, 
                                         malig_ukbb$Chemical_workplace.0.12, malig_ukbb$Chemical_workplace.0.13, malig_ukbb$Chemical_workplace.0.14, 
                                         malig_ukbb$Chemical_workplace.0.15, malig_ukbb$Chemical_workplace.0.16, 
                                         malig_ukbb$Chemical_workplace.0.17, malig_ukbb$Chemical_workplace.0.18, malig_ukbb$Chemical_workplace.0.19, 
                                         malig_ukbb$Chemical_workplace.0.20, malig_ukbb$Chemical_workplace.0.21, 
                                         malig_ukbb$Chemical_workplace.0.22, malig_ukbb$Chemical_workplace.0.23, malig_ukbb$Chemical_workplace.0.24, 
                                         malig_ukbb$Chemical_workplace.0.25, malig_ukbb$Chemical_workplace.0.26, 
                                         malig_ukbb$Chemical_workplace.0.27, malig_ukbb$Chemical_workplace.0.28, 
                                         malig_ukbb$Chemical_workplace.0.29, malig_ukbb$Chemical_workplace.0.30, malig_ukbb$Chemical_workplace.0.31, 
                                         malig_ukbb$Chemical_workplace.0.32, malig_ukbb$Chemical_workplace.0.33, 
                                         malig_ukbb$Chemical_workplace.0.34, malig_ukbb$Chemical_workplace.0.35, malig_ukbb$Chemical_workplace.0.36, 
                                         malig_ukbb$Chemical_workplace.0.37, malig_ukbb$Chemical_workplace.0.38, 
                                         malig_ukbb$Chemical_workplace.0.39)



malig_ukbb$Diesel_exhaust_workplace<- coalesce(malig_ukbb$Diesel_exhaust_workplace.0.0, malig_ukbb$Diesel_exhaust_workplace.0.1, 
                                               malig_ukbb$Diesel_exhaust_workplace.0.2, malig_ukbb$Diesel_exhaust_workplace.0.3, 
                                               malig_ukbb$Diesel_exhaust_workplace.0.4, malig_ukbb$Diesel_exhaust_workplace.0.5, malig_ukbb$Diesel_exhaust_workplace.0.6, 
                                               malig_ukbb$Diesel_exhaust_workplace.0.7, malig_ukbb$Diesel_exhaust_workplace.0.8, malig_ukbb$Diesel_exhaust_workplace.0.9, 
                                               malig_ukbb$Diesel_exhaust_workplace.0.10, malig_ukbb$Diesel_exhaust_workplace.0.11, 
                                               malig_ukbb$Diesel_exhaust_workplace.0.12, malig_ukbb$Diesel_exhaust_workplace.0.13, malig_ukbb$Diesel_exhaust_workplace.0.14, 
                                               malig_ukbb$Diesel_exhaust_workplace.0.15, malig_ukbb$Diesel_exhaust_workplace.0.16, 
                                               malig_ukbb$Diesel_exhaust_workplace.0.17, malig_ukbb$Diesel_exhaust_workplace.0.18, malig_ukbb$Diesel_exhaust_workplace.0.19, 
                                               malig_ukbb$Diesel_exhaust_workplace.0.20, malig_ukbb$Diesel_exhaust_workplace.0.21, 
                                               malig_ukbb$Diesel_exhaust_workplace.0.22, malig_ukbb$Diesel_exhaust_workplace.0.23, malig_ukbb$Diesel_exhaust_workplace.0.24, 
                                               malig_ukbb$Diesel_exhaust_workplace.0.25, malig_ukbb$Diesel_exhaust_workplace.0.26, 
                                               malig_ukbb$Diesel_exhaust_workplace.0.27, malig_ukbb$Diesel_exhaust_workplace.0.28, 
                                               malig_ukbb$Diesel_exhaust_workplace.0.29, malig_ukbb$Diesel_exhaust_workplace.0.30, malig_ukbb$Diesel_exhaust_workplace.0.31, 
                                               malig_ukbb$Diesel_exhaust_workplace.0.32, malig_ukbb$Diesel_exhaust_workplace.0.33, 
                                               malig_ukbb$Diesel_exhaust_workplace.0.34, malig_ukbb$Diesel_exhaust_workplace.0.35, malig_ukbb$Diesel_exhaust_workplace.0.36, 
                                               malig_ukbb$Diesel_exhaust_workplace.0.37, malig_ukbb$Diesel_exhaust_workplace.0.38, 
                                               malig_ukbb$Diesel_exhaust_workplace.0.39)

malig_ukbb$Asbestos_workplace<- coalesce(malig_ukbb$Asbestos_workplace.0.0, malig_ukbb$Asbestos_workplace.0.1, 
                                         malig_ukbb$Asbestos_workplace.0.2, malig_ukbb$Asbestos_workplace.0.3, 
                                         malig_ukbb$Asbestos_workplace.0.4, malig_ukbb$Asbestos_workplace.0.5, malig_ukbb$Asbestos_workplace.0.6, 
                                         malig_ukbb$Asbestos_workplace.0.7, malig_ukbb$Asbestos_workplace.0.8, malig_ukbb$Asbestos_workplace.0.9, 
                                         malig_ukbb$Asbestos_workplace.0.10, malig_ukbb$Asbestos_workplace.0.11, 
                                         malig_ukbb$Asbestos_workplace.0.12, malig_ukbb$Asbestos_workplace.0.13, malig_ukbb$Asbestos_workplace.0.14, 
                                         malig_ukbb$Asbestos_workplace.0.15, malig_ukbb$Asbestos_workplace.0.16, 
                                         malig_ukbb$Asbestos_workplace.0.17, malig_ukbb$Asbestos_workplace.0.18, malig_ukbb$Asbestos_workplace.0.19, 
                                         malig_ukbb$Asbestos_workplace.0.20, malig_ukbb$Asbestos_workplace.0.21, 
                                         malig_ukbb$Asbestos_workplace.0.22, malig_ukbb$Asbestos_workplace.0.23, malig_ukbb$Asbestos_workplace.0.24, 
                                         malig_ukbb$Asbestos_workplace.0.25, malig_ukbb$Asbestos_workplace.0.26, 
                                         malig_ukbb$Asbestos_workplace.0.27, malig_ukbb$Asbestos_workplace.0.28, 
                                         malig_ukbb$Asbestos_workplace.0.29, malig_ukbb$Asbestos_workplace.0.30, malig_ukbb$Asbestos_workplace.0.31, 
                                         malig_ukbb$Asbestos_workplace.0.32, malig_ukbb$Asbestos_workplace.0.33, 
                                         malig_ukbb$Asbestos_workplace.0.34, malig_ukbb$Asbestos_workplace.0.35, malig_ukbb$Asbestos_workplace.0.36, 
                                         malig_ukbb$Asbestos_workplace.0.37, malig_ukbb$Asbestos_workplace.0.38, 
                                         malig_ukbb$Asbestos_workplace.0.39)

malig_ukbb$Paint_glue_thinner_workplace <- coalesce(malig_ukbb$Paint_glue_thinner_workplace.0.0, malig_ukbb$Paint_glue_thinner_workplace.0.1, 
                                                    malig_ukbb$Paint_glue_thinner_workplace.0.2, malig_ukbb$Paint_glue_thinner_workplace.0.3, 
                                                    malig_ukbb$Paint_glue_thinner_workplace.0.4, malig_ukbb$Paint_glue_thinner_workplace.0.5, malig_ukbb$Paint_glue_thinner_workplace.0.6, 
                                                    malig_ukbb$Paint_glue_thinner_workplace.0.7, malig_ukbb$Paint_glue_thinner_workplace.0.8, malig_ukbb$Paint_glue_thinner_workplace.0.9, 
                                                    malig_ukbb$Paint_glue_thinner_workplace.0.10, malig_ukbb$Paint_glue_thinner_workplace.0.11, 
                                                    malig_ukbb$Paint_glue_thinner_workplace.0.12, malig_ukbb$Paint_glue_thinner_workplace.0.13, malig_ukbb$Asbestos_workplace.0.14, 
                                                    malig_ukbb$Paint_glue_thinner_workplace.0.15, malig_ukbb$Paint_glue_thinner_workplace.0.16, 
                                                    malig_ukbb$Paint_glue_thinner_workplace.0.17, malig_ukbb$Paint_glue_thinner_workplace.0.18, malig_ukbb$Asbestos_workplace.0.19, 
                                                    malig_ukbb$Paint_glue_thinner_workplace.0.20, malig_ukbb$Paint_glue_thinner_workplace.0.21, 
                                                    malig_ukbb$Paint_glue_thinner_workplace.0.22, malig_ukbb$Paint_glue_thinner_workplace.0.23, malig_ukbb$Asbestos_workplace.0.24, 
                                                    malig_ukbb$Paint_glue_thinner_workplace.0.25, malig_ukbb$Paint_glue_thinner_workplace.0.26, 
                                                    malig_ukbb$Paint_glue_thinner_workplace.0.27, malig_ukbb$Paint_glue_thinner_workplace.0.28, 
                                                    malig_ukbb$Paint_glue_thinner_workplace.0.29, malig_ukbb$Paint_glue_thinner_workplace.0.30, malig_ukbb$Paint_glue_thinner_workplace.0.31, 
                                                    malig_ukbb$Paint_glue_thinner_workplace.0.32, malig_ukbb$Paint_glue_thinner_workplace.0.33, 
                                                    malig_ukbb$Paint_glue_thinner_workplace.0.34, malig_ukbb$Paint_glue_thinner_workplace.0.35, malig_ukbb$Paint_glue_thinner_workplace.0.36, 
                                                    malig_ukbb$Paint_glue_thinner_workplace.0.37, malig_ukbb$Paint_glue_thinner_workplace.0.38, 
                                                    malig_ukbb$Paint_glue_thinner_workplace.0.39)

malig_ukbb$Pesticide_workplace <- coalesce(malig_ukbb$Pesticide_workplace.0.0, malig_ukbb$Pesticide_workplace.0.1, 
                                           malig_ukbb$Pesticide_workplace.0.2, malig_ukbb$Pesticide_workplace.0.3, 
                                           malig_ukbb$Pesticide_workplace.0.4, malig_ukbb$Pesticide_workplace.0.5, malig_ukbb$Pesticide_workplace.0.6, malig_ukbb$Pesticide_workplace.0.7,
                                           malig_ukbb$Pesticide_workplace.0.8, malig_ukbb$Pesticide_workplace.0.9, 
                                           malig_ukbb$Pesticide_workplace.0.10, malig_ukbb$Pesticide_workplace.0.11, 
                                           malig_ukbb$Pesticide_workplace.0.12, malig_ukbb$Pesticide_workplace.0.13, malig_ukbb$Pesticide_workplace.0.14, 
                                           malig_ukbb$Pesticide_workplace.0.15, malig_ukbb$Pesticide_workplace.0.16, 
                                           malig_ukbb$Pesticide_workplace.0.17, malig_ukbb$Pesticide_workplace.0.18, malig_ukbb$Pesticide_workplace.0.19, 
                                           malig_ukbb$Pesticide_workplace.0.20, malig_ukbb$Pesticide_workplace.0.21, 
                                           malig_ukbb$Pesticide_workplace.0.22, malig_ukbb$Pesticide_workplace.0.23, malig_ukbb$Pesticide_workplace.0.24, 
                                           malig_ukbb$Pesticide_workplace.0.25, malig_ukbb$Pesticide_workplace.0.26, 
                                           malig_ukbb$Pesticide_workplace.0.27, malig_ukbb$Pesticide_workplace.0.28, 
                                           malig_ukbb$Pesticide_workplace.0.29, malig_ukbb$Pesticide_workplace.0.30, malig_ukbb$Pesticide_workplace.0.31, 
                                           malig_ukbb$Pesticide_workplace.0.32, malig_ukbb$Pesticide_workplace.0.33, 
                                           malig_ukbb$Pesticide_workplace.0.34, malig_ukbb$Pesticide_workplace.0.35, malig_ukbb$Pesticide_workplace.0.36, 
                                           malig_ukbb$Pesticide_workplace.0.37, malig_ukbb$Pesticide_workplace.0.38, 
                                           malig_ukbb$Pesticide_workplace.0.39)

table(malig_ukbb$PPesticide_workplace.0.25)
malig_ukbb$Dusty_workplace <- coalesce(malig_ukbb$Dusty_workplace.0.0, malig_ukbb$Dusty_workplace.0.1, 
                                       malig_ukbb$Dusty_workplace.0.2, malig_ukbb$Dusty_workplace.0.3, 
                                       malig_ukbb$Dusty_workplace.0.4, malig_ukbb$Dusty_workplace.0.5, malig_ukbb$Dusty_workplace.0.6, 
                                       malig_ukbb$Dusty_workplace.0.7, malig_ukbb$Dusty_workplace.0.8, malig_ukbb$Dusty_workplace.0.9, 
                                       malig_ukbb$Dusty_workplace.0.10, malig_ukbb$Dusty_workplace.0.11, 
                                       malig_ukbb$Dusty_workplace.0.12, malig_ukbb$Dusty_workplace.0.13, malig_ukbb$Dusty_workplace.0.14, 
                                       malig_ukbb$Dusty_workplace.0.15, malig_ukbb$Dusty_workplace.0.16, 
                                       malig_ukbb$Dusty_workplace.0.17, malig_ukbb$Dusty_workplace.0.18, malig_ukbb$Dusty_workplace.0.19, 
                                       malig_ukbb$Dusty_workplace.0.20, malig_ukbb$Dusty_workplace.0.21, 
                                       malig_ukbb$Dusty_workplace.0.22, malig_ukbb$Dusty_workplace.0.23, malig_ukbb$Dusty_workplace.0.24, 
                                       malig_ukbb$Dusty_workplace.0.25, malig_ukbb$Dusty_workplace.0.26, 
                                       malig_ukbb$Dusty_workplace.0.27, malig_ukbb$Dusty_workplace.0.28, 
                                       malig_ukbb$Dusty_workplace.0.29, malig_ukbb$Dusty_workplace.0.30, malig_ukbb$Dusty_workplace.0.31, 
                                       malig_ukbb$Dusty_workplace.0.32, malig_ukbb$Dusty_workplace.0.33, 
                                       malig_ukbb$Dusty_workplace.0.34, malig_ukbb$Dusty_workplace.0.35, malig_ukbb$Dusty_workplace.0.36, 
                                       malig_ukbb$Dusty_workplace.0.37, malig_ukbb$Dusty_workplace.0.38, 
                                       malig_ukbb$Dusty_workplace.0.39)


# Drop those with prevalent blood cancer /disorder
malig_ukbbd<-subset(malig_ukbb, prevalent_case!=1)

datafinal <- malig_ukbbd

blood_data <- datafinal %>% dplyr::select(Medication_cholesterol, Alcohol_intake_yesterday, Alcohol_consumed, Alcohol_intake_frequency..0.0, Alcohol_drink_status.0.0, Current_tobacco_smoking.0.0, Time_since_last_menstrual_period,
                                          Sex.0.0, YOB.0.0, eid, UK_Biobank_assessment_centre.0.0, 
                                          Leukocyte_count.0.0, Erythrocyte_count.0.0, Had_menopause.0.0, 
                                          Haematocrit_percentage.0.0, Mean_corpuscular_volume.0.0,
                                          Mean_corpuscular_haemoglobin_conc.0.0, Erythrocyte_distribution_width.0.0,
                                          Platelet_count.0.0, Platelet_crit.0.0, Thrombocyte_volume.0.0, Platelet_distribution_width.0.0,
                                          Lymphocyte_count.0.0, Monocyte_count.0.0, Neutrophill_count.0.0, Eosinophil_count.0.0, 
                                          Basophill_count.0.0, Lymphocyte_percentage.0.0,
                                          Monocyte_percentage.0.0, Neutrophill_percentage.0.0, Eosinophill_percentage.0.0,
                                          Basophill_percentage.0.0, Nucleated_red_blood_cell_percentage.0.0, Reticulocyte_percentage.0.0,
                                          Reticulocyte_count.0.0, Mean_reticulocyte_volume.0.0, Mean_sphered_cell_volume.0.0,
                                          Immature_reticulocyte_fraction.0.0, High_light_scatter_reticulocyte_percentage.0.0, 
                                          High_light_scatter_reticulocyte_count.0.0, Smoking_status.0.0,
                                          Cystatin_C_2.0.0,
                                          Phosphate_2.0.0,
                                          Aspartate_aminotransferase_2.0.0,
                                          Glycated_haemoglobin_HbA1c_2.0.0,
                                          Apolipoprotein_A_2.0.0,
                                          Apolipoprotein_B_2.0.0,
                                          HDL_cholesterol_2.0.0,
                                          LDL_direct_2.0.0,
                                          Cholesterol_2.0.0,
                                          Pack_years_smoking.0.0,
                                          BMI.0.0,
                                          No_cancers.0.0,
                                          Usual_walking_pace.0.0,
                                          Processed_meat_intake.0.0,
                                          Maternal_smoking_around_birth.0.0,
                                          Use_of_sun_uv_protection.0.0,
                                          Frequency_of_other_exercises_in_last_4_weeks.0.0,
                                          Menstruating_today.0.0,
                                          War_exposure.0.0,
                                          Ever_suicide.0.0,
                                          Ever_addicted_alcohol.0.0,
                                          Ethnicity.0.0,
                                          Sexually_molested_child.0.0,
                                          Length_of_mobile_phone_use.0.0, 
                                          Nap_during_day.0.0,                                                            
                                          Oily_fish_intake.0.0,Processed_meat_intake.0.0,Poultry_intake.0.0, Beef_intake.0.0, Pork_intake.0.0, Lamb_mutton_intake.0.0,
                                          Salt_added_to_food.0.0, Skin_colour.0.0, Facial_ageing.0.0, Mood_swings.0.0,
                                          Risk_taking.0.0, Frequency_of_depressed_mood_in_last_2_weeks.0.0, Miserableness.0.0, Able_to_confide.0.0,
                                          Chest_pain_or_discomfort.0.0,
                                          Noisy_workplace.0.0,War_exposure.0.0,
                                          Cough_most_days.0.0,Dusty_workplace,Pesticide_workplace,Paint_glue_thinner_workplace,
                                          Asbestos_workplace,Diesel_exhaust_workplace,Chemical_workplace)

names(blood_data) = gsub(pattern = "_2.0.0*", replacement = "", x = names(blood_data))
names(blood_data) = gsub(pattern = ".0.0*", replacement = "", x = names(blood_data))

table(blood_data$Poultry_intake.0.0)
table(blood_data$Beef_intake.0.0)
table(blood_data$Pork_intake.0.0)
table(blood_data$Lamb_mutton_intake.0.0)
table(blood_data$Salt_added_to_food.0.0)
table(blood_data$Skin_colour.0.0)
table(blood_data$Facial_ageing.0.0)
table(blood_data$Mood_swings.0.0)
table(blood_data$Risk_taking.0.0)
table(blood_data$Frequency_of_depressed_mood_in_last_2_weeks.0.0)
table(blood_data$Miserableness.0.0)
table(blood_data$Able_to_confide.0.0)
table(blood_data$Chest_pain_or_discomfort.0.0)
table(blood_data$Noisy_workplace.0.0)
table(blood_data$IBS.0.0)
table(blood_data$Chronic_bronchitis.0.0)
table(blood_data$Asbestosis.0.0)
table(blood_data$Cough_most_days.0.0)
table(blood_data$Dusty_workplace)
table(blood_data$Pesticide_workplace)
table(blood_data$Dusty_workplace)
table(blood_data$Paint_glue_thinner_workplace)
table(blood_data$Asbestos_workplace)
table(blood_data$Diesel_exhaust_workplace)
table(blood_data$Chemical_workplace)

## Select variables from George's paper
Blood_final <- blood_data %>% dplyr::select(-Sexually_molested_child, -Menstruating_today)

setwd("/rds/general/user/iw413/home/Summerproject/outputs")

#Merge the datasets
fam=read.table("/rds/general/project/chadeau_ukbb_folder/live/data/project_data/Genetics/ukb22418_c17_b0_v2_s488175.fam",he=F, strings=F)
fam = fam[-1,]
fam2=read.table("ukb22418_c1_b0_v2_s488248.fam",he=T, strings=F)
warnings()
setwd("/rds/general/user/iw413/home/Summerproject/outputs")
ch=read.table("results_var_ch_all_allgenes_UKB.txt",he=T, strings=F) # read the new file that Pedro sent here
fam2 <- fam2 %>% 
  rename(
    eid = X2039051)
fam2$order <- row.names(fam2) 
fam2$order <- as.integer(fam2$order)
conflict_prefer("count", "dplyr")
mergedchfam2 <- merge(x = fam2, y = ch, by = "eid", all.x = TRUE)

## Investigating duplicates and deleting duplicates FOR NOW!
##n_occur <- data.frame(table(mergedchfam2$eid))
##view <- n_occur[n_occur$Freq > 2,]
##view <- mergedchfam2[mergedchfam2$eid %in% n_occur$Var1[n_occur$Freq > 1],]
##duplicated(view$eid)

mergedchfam22 <- mergedchfam2[!duplicated(mergedchfam2$eid), ]
ch2 <- mergedchfam22
ch2 <- ch2[order(ch2$order, decreasing = FALSE),]  
head(fam)
head(ch2)
dim(fam) # should be the same dimension as below
dim(ch2)
dat=cbind(fam,ch2) # combine the two datasets
head(dat)
table(dat$V5, dat$X1) # cross-check that most sex data is the same
Blood_finalised <- merge(Blood_final, dat, by.x="eid", by.y="V1")

##Delete variables from CH dataset

Blood_finalised <- Blood_finalised %>% dplyr::select(-X2039051.1, -X0, -X0.1, -X1, -order, -ukb_id, -chr, -pos, -ref_allele, -alt_allele, -mutect2_filt,
                                                     -consequence, -transcript, - alt_ad, -ref_ad, -V2, -V3, -V4, -V5, -V6, -cds_position, -protein_position, -amino_acids,
                                                     -codons, -c_mut, -p_mut, -hgvsp, -hgvsc, -exon, -variant_class, -impact, -existing_variation)

##Make factors

Blood_finalised$Sex <- as.factor(Blood_finalised$Sex)
Blood_finalised$symbol <- as.factor(Blood_finalised$symbol)

##Remove those with no age
Blood_finalised <- Blood_finalised %>% mutate(age = 2023 - Blood_finalised$YOB)
Blood_finalised <- Blood_finalised[!is.na(Blood_finalised$age), ]

##Defining large clone variable
Blood_finalised$largeclone01 <- ifelse(Blood_finalised$vaf >= 0.1, "large_clone" , Blood_finalised$vaf)
Blood_finalised$largeclone01 <- ifelse(Blood_finalised$vaf < 0.1, "small_clone" , Blood_finalised$largeclone01)
Blood_finalised$largeclone01[is.na(Blood_finalised$largeclone01)] = "no_mutation"

##Defining large clone variable
Blood_finalised$largeclone015 <- ifelse(Blood_finalised$vaf >= 0.15, "large_clone" , Blood_finalised$vaf)
Blood_finalised$largeclone015 <- ifelse(Blood_finalised$vaf < 0.15, "small_clone" , Blood_finalised$largeclone01)
Blood_finalised$largeclone015[is.na(Blood_finalised$largeclone015)] = "no_mutation"

##Defining very large clone variable
Blood_finalised$largeclone02 <- ifelse(Blood_finalised$vaf >= 0.2, " large_clone" , Blood_finalised$vaf)
Blood_finalised$largeclone02 <- ifelse(Blood_finalised$vaf < 0.2, "small_clone" , Blood_finalised$largeclone02)
Blood_finalised$largeclone02[is.na(Blood_finalised$largeclone02)] = "no_mutation"

##Defining clone variable
Blood_finalised$clone <- ifelse(Blood_finalised$vaf > 0.0, "clone" , Blood_finalised$vaf)
Blood_finalised$clone[is.na(Blood_finalised$clone)] = "no_mutation"
Blood_finaliseds <- Blood_finalised

levels(Blood_finaliseds$Ever_addicted_alcohol)

Blood_finaliseds<- Blood_finalised %>%mutate(Ethnicity= recode(Ethnicity,
                                                               "Prefer not to answer"="Unknown",
                                                               "Do not know"="Unknown",
                                                               "White"="White/White British",
                                                               "British"="White/White British",
                                                               "Irish"="White/White British",
                                                               "Any other white background"="White/White British",
                                                               "Mixed"="Mixed",
                                                               "White and Black Caribbean"="Mixed",
                                                               "White and Black African"="Mixed",
                                                               "White and Asian"="Mixed",
                                                               "Any other mixed background"="Mixed",
                                                               "Asian or Asian British"="Asian/Asian British",
                                                               "Indian"="Asian/Asian British",
                                                               "Pakistani"="Asian/Asian British",
                                                               "Bangladeshi"="Asian/Asian British",
                                                               "Any other Asian background"="Asian/Asian British",
                                                               "Black or Black British"="Black/Black British",
                                                               "Caribbean" = "Black/Black British",
                                                               "African"="Black/Black British",
                                                               "Any other Black background"="Black/Black British",
                                                               "Chinese"="Chinese",
                                                               "Other ethnic group"="Other"),
                                             
                                             
                                             Had_menopause= recode(Had_menopause, "Prefer not to answer"="Unknown",
                                                                   "Not sure - had a hysterectomy"="Unknown_hysterectomy",
                                                                   "Not sure - other reason"="Unknown"),
                                             Alcohol_intake_frequency.= recode(Alcohol_intake_frequency., "Prefer not to answer"="Unknown"),
                                             Smoking_status= recode(Smoking_status, "Prefer not to answer"="Unknown"),
                                             Usual_walking_pace= recode(Usual_walking_pace, "Prefer not to answer"="Unknown"),
                                             Processed_meat_intake= recode(Processed_meat_intake, "Prefer not to answer"="Unknown"),
                                             Processed_meat_intake= recode(Processed_meat_intake, "Do not know"="Unknown"),
                                             Alcohol_drink_status= recode(Alcohol_drink_status, "Prefer not to answer"="Unknown"),
                                             Alcohol_drink_status= recode(Alcohol_drink_status, "Prefer not to answer"="Unknown"),
                                             Use_of_sun_uv_protection= recode(Use_of_sun_uv_protection, "Prefer not to answer"="Unknown"),
                                             Use_of_sun_uv_protection= recode(Use_of_sun_uv_protection, "Do not know"="Unknown"),
                                             Maternal_smoking_around_birth= recode(Maternal_smoking_around_birth, "Do not know"="Unknown"),
                                             Current_tobacco_smoking= recode(Current_tobacco_smoking, "Prefer not to answer"="Unknown"),
                                             Frequency_of_other_exercises_in_last_4_weeks= recode(Frequency_of_other_exercises_in_last_4_weeks, "Prefer not to answer"="Unknown"),
                                             Frequency_of_other_exercises_in_last_4_weeks= recode(Frequency_of_other_exercises_in_last_4_weeks, "Do not know"="Unknown"),
                                             War_exposure= recode(War_exposure, "Prefer not to answer"="Unknown"),
                                             War_exposure= recode(War_exposure, "Yes, but not in the last 12 months"="Yes"),
                                             War_exposure= recode(War_exposure, "Yes, within the last 12 months"="Yes"),
                                             Ever_suicide= recode(Ever_suicide, "Prefer not to answer"="Unknown"),
                                             Ever_addicted_alcohol= recode(Ever_addicted_alcohol, "Prefer not to answer"="Unknown"),
                                             Ever_addicted_alcohol= recode(Ever_addicted_alcohol, "Do not know"="Unknown"),
                                             Ever_addicted_alcohol= recode(Ever_addicted_alcohol, "No"="No"),
                                             Length_of_mobile_phone_use= recode(Length_of_mobile_phone_use, "Do not know"="Unknown"),
                                             Length_of_mobile_phone_use= recode(Length_of_mobile_phone_use, "Prefer not to answer"="Unknown"),
                                             Nap_during_day= recode(Nap_during_day, "Prefer not to answer"="Unknown"),
                                             Oily_fish_intake= recode(Oily_fish_intake, "Prefer not to answer"="Unknown",
                                                                      "Do not know"="Unknown",
                                                                      "2-4 times a week"="More_than_2_times_week",
                                                                      "5-6 times a week"="More_than_2_times_week",
                                                                      "Once or more daily"="More_than_2_times_week"),
                                             Processed_meat_intake= recode(Processed_meat_intake, "Prefer not to answer"="Unknown",
                                                                           "Do not know"="Unknown",
                                                                           "2-4 times a week"="More_than_2_times_week",
                                                                           "5-6 times a week"="More_than_2_times_week",
                                                                           "Once or more daily"="More_than_2_times_week"),
                                             Poultry_intake= recode(Poultry_intake, "Prefer not to answer"="Unknown",
                                                                    "Do not know"="Unknown",
                                                                    "2-4 times a week"="More_than_2_times_week",
                                                                    "5-6 times a week"="More_than_2_times_week",
                                                                    "Once or more daily"="More_than_2_times_week"),
                                             Beef_intake= recode(Beef_intake, "Prefer not to answer"="Unknown",
                                                                 "Do not know"="Unknown",
                                                                 "2-4 times a week"="More_than_2_times_week",
                                                                 "5-6 times a week"="More_than_2_times_week",
                                                                 "Once or more daily"="More_than_2_times_week"),
                                             Pork_intake= recode(Pork_intake, "Prefer not to answer"="Unknown",
                                                                 "Do not know"="Unknown",
                                                                 "2-4 times a week"="More_than_2_times_week",
                                                                 "5-6 times a week"="More_than_2_times_week",
                                                                 "Once or more daily"="More_than_2_times_week"),
                                             Lamb_mutton_intake= recode(Lamb_mutton_intake, "Prefer not to answer"="Unknown",
                                                                        "Do not know"="Unknown",
                                                                        "2-4 times a week"="More_than_2_times_week",
                                                                        "5-6 times a week"="More_than_2_times_week",
                                                                        "Once or more daily"="More_than_2_times_week"),
                                             Salt_added_to_food= recode(Salt_added_to_food, "Prefer not to answer"="Unknown"),
                                             Skin_colour= recode(Skin_colour, "Prefer not to answer"="Unknown",
                                                                 "Do not know"="Unknown"),
                                             Facial_ageing= recode(Facial_ageing, "Prefer not to answer"="Unknown",
                                                                   "Do not know"="Unknown"),
                                             Mood_swings= recode(Mood_swings, "Prefer not to answer"="Unknown",
                                                                 "Do not know"="Unknown"),
                                             Risk_taking= recode(Risk_taking, "Prefer not to answer"="Unknown",
                                                                 "Do not know"="Unknown"),
                                             Frequency_of_depressed_mood_in_last_2_weeks= recode(Frequency_of_depressed_mood_in_last_2_weeks, "Prefer not to answer"="Unknown",
                                                                                                 "Do not know"="Unknown"),
                                             Miserableness= recode(Miserableness, "Prefer not to answer"="Unknown",
                                                                   "Do not know"="Unknown"),
                                             Chest_pain_or_discomfort= recode(Chest_pain_or_discomfort, "Prefer not to answer"="Unknown",
                                                                              "Do not know"="Unknown"),
                                             Noisy_workplace= recode(Noisy_workplace, "Prefer not to answer"="Unknown",
                                                                     "Do not know"="Unknown"),
                                             Dusty_workplace= recode(Dusty_workplace,
                                                                     "Do not know"="Unknown"),
                                             Pesticide_workplace= recode(Pesticide_workplace,
                                                                         "Do not know"="Unknown"),
                                             Paint_glue_thinner_workplace= recode(Paint_glue_thinner_workplace,
                                                                                  "Do not know"="Unknown"),
                                             Asbestos_workplace= recode(Asbestos_workplace,
                                                                        "Do not know"="Unknown"),
                                             Diesel_exhaust_workplace= recode(Diesel_exhaust_workplace,
                                                                              "Do not know"="Unknown"),
                                             Chemical_workplace= recode(Chemical_workplace,
                                                                        "Do not know"="Unknown"))



Blood_finaliseds$Poultry_intake <- as.character(Blood_finaliseds$Poultry_intake)
Blood_finaliseds$Poultry_intake[is.na(Blood_finaliseds$Poultry_intake)] <- "Unknown"
Blood_finaliseds$Poultry_intake <- as.factor(Blood_finaliseds$Poultry_intake)

Blood_finaliseds$Beef_intake <- as.character(Blood_finaliseds$Beef_intake)
Blood_finaliseds$Beef_intake[is.na(Blood_finaliseds$Beef_intake)] <- "Unknown"
Blood_finaliseds$Beef_intake <- as.factor(Blood_finaliseds$Beef_intake)

Blood_finaliseds$Pork_intake <- as.character(Blood_finaliseds$Pork_intake)
Blood_finaliseds$Pork_intake[is.na(Blood_finaliseds$Pork_intake)] <- "Unknown"
Blood_finaliseds$Pork_intake <- as.factor(Blood_finaliseds$Pork_intake)

Blood_finaliseds$Lamb_mutton_intake <- as.character(Blood_finaliseds$Lamb_mutton_intake)
Blood_finaliseds$Lamb_mutton_intake[is.na(Blood_finaliseds$Lamb_mutton_intake)] <- "Unknown"
Blood_finaliseds$Lamb_mutton_intake <- as.factor(Blood_finaliseds$Lamb_mutton_intake)

Blood_finaliseds$Salt_added_to_food <- as.character(Blood_finaliseds$Salt_added_to_food)
Blood_finaliseds$Salt_added_to_food[is.na(Blood_finaliseds$Salt_added_to_food)] <- "Unknown"
Blood_finaliseds$Salt_added_to_food <- as.factor(Blood_finaliseds$Salt_added_to_food)

Blood_finaliseds$Skin_colour <- as.character(Blood_finaliseds$Skin_colour)
Blood_finaliseds$Skin_colour[is.na(Blood_finaliseds$Skin_colour)] <- "Unknown"
Blood_finaliseds$Skin_colour <- as.factor(Blood_finaliseds$Skin_colour)

Blood_finaliseds$Facial_ageing <- as.character(Blood_finaliseds$Facial_ageing)
Blood_finaliseds$Facial_ageing[is.na(Blood_finaliseds$Facial_ageing)] <- "Unknown"
Blood_finaliseds$Facial_ageing <- as.factor(Blood_finaliseds$Facial_ageing)

Blood_finaliseds$Mood_swings <- as.character(Blood_finaliseds$Mood_swings)
Blood_finaliseds$Mood_swings[is.na(Blood_finaliseds$Mood_swings)] <- "Unknown"
Blood_finaliseds$Mood_swings <- as.factor(Blood_finaliseds$Mood_swings)

Blood_finaliseds$Risk_taking <- as.character(Blood_finaliseds$Risk_taking)
Blood_finaliseds$Risk_taking[is.na(Blood_finaliseds$Risk_taking)] <- "Unknown"
Blood_finaliseds$Risk_taking <- as.factor(Blood_finaliseds$Risk_taking)

Blood_finaliseds$Frequency_of_depressed_mood_in_last_2_weeks <- as.character(Blood_finaliseds$Frequency_of_depressed_mood_in_last_2_weeks)
Blood_finaliseds$Frequency_of_depressed_mood_in_last_2_weeks[is.na(Blood_finaliseds$Frequency_of_depressed_mood_in_last_2_weeks)] <- "Unknown"
Blood_finaliseds$Frequency_of_depressed_mood_in_last_2_weeks <- as.factor(Blood_finaliseds$Frequency_of_depressed_mood_in_last_2_weeks)

Blood_finaliseds$Miserableness <- as.character(Blood_finaliseds$Miserableness)
Blood_finaliseds$Miserableness[is.na(Blood_finaliseds$Miserableness)] <- "Unknown"
Blood_finaliseds$Miserableness <- as.factor(Blood_finaliseds$Miserableness)

Blood_finaliseds$Able_to_confide <- as.character(Blood_finaliseds$Able_to_confide)
Blood_finaliseds$Able_to_confide[is.na(Blood_finaliseds$Able_to_confide)] <- "Unknown"
Blood_finaliseds$Able_to_confide <- as.factor(Blood_finaliseds$Able_to_confide)

Blood_finaliseds$Chest_pain_or_discomfort <- as.character(Blood_finaliseds$Chest_pain_or_discomfort)
Blood_finaliseds$Chest_pain_or_discomfort[is.na(Blood_finaliseds$Chest_pain_or_discomfort)] <- "Unknown"
Blood_finaliseds$Chest_pain_or_discomfort <- as.factor(Blood_finaliseds$Chest_pain_or_discomfort)

Blood_finaliseds$Noisy_workplace <- as.character(Blood_finaliseds$Noisy_workplace)
Blood_finaliseds$Noisy_workplace[is.na(Blood_finaliseds$Noisy_workplace)] <- "Unknown"
Blood_finaliseds$Noisy_workplace <- as.factor(Blood_finaliseds$Noisy_workplace)

Blood_finaliseds$Cough_most_days <- as.character(Blood_finaliseds$Cough_most_days)
Blood_finaliseds$Cough_most_days[is.na(Blood_finaliseds$Cough_most_days)] <- "Unknown"
Blood_finaliseds$Cough_most_days <- as.factor(Blood_finaliseds$Cough_most_days)

Blood_finaliseds$Pesticide_workplace <- as.character(Blood_finaliseds$Pesticide_workplace)
Blood_finaliseds$Pesticide_workplace[is.na(Blood_finaliseds$Pesticide_workplace)] <- "Unknown"
Blood_finaliseds$Pesticide_workplace <- as.factor(Blood_finaliseds$Pesticide_workplace)

Blood_finaliseds$Dusty_workplace <- as.character(Blood_finaliseds$Dusty_workplace)
Blood_finaliseds$Dusty_workplace[is.na(Blood_finaliseds$Dusty_workplace)] <- "Unknown"
Blood_finaliseds$Dusty_workplace <- as.factor(Blood_finaliseds$Dusty_workplace)

Blood_finaliseds$Paint_glue_thinner_workplace <- as.character(Blood_finaliseds$Paint_glue_thinner_workplace)
Blood_finaliseds$Paint_glue_thinner_workplace[is.na(Blood_finaliseds$Paint_glue_thinner_workplace)] <- "Unknown"
Blood_finaliseds$Paint_glue_thinner_workplace <- as.factor(Blood_finaliseds$Paint_glue_thinner_workplace)

Blood_finaliseds$Asbestos_workplace <- as.character(Blood_finaliseds$Asbestos_workplace)
Blood_finaliseds$Asbestos_workplace[is.na(Blood_finaliseds$Asbestos_workplace)] <- "Unknown"
Blood_finaliseds$Asbestos_workplace <- as.factor(Blood_finaliseds$Asbestos_workplace)

Blood_finaliseds$Diesel_exhaust_workplace <- as.character(Blood_finaliseds$Diesel_exhaust_workplace)
Blood_finaliseds$Diesel_exhaust_workplace[is.na(Blood_finaliseds$Diesel_exhaust_workplace)] <- "Unknown"
Blood_finaliseds$Diesel_exhaust_workplace <- as.factor(Blood_finaliseds$Diesel_exhaust_workplace)

Blood_finaliseds$Chemical_workplace <- as.character(Blood_finaliseds$Chemical_workplace)
Blood_finaliseds$Chemical_workplace[is.na(Blood_finaliseds$Chemical_workplace)] <- "Unknown"
Blood_finaliseds$Chemical_workplace <- as.factor(Blood_finaliseds$Chemical_workplace)

Blood_finaliseds$Oily_fish_intake <- as.character(Blood_finaliseds$Oily_fish_intake)
Blood_finaliseds$Oily_fish_intake[is.na(Blood_finaliseds$Oily_fish_intake)] <- "Unknown"
Blood_finaliseds$Oily_fish_intake <- as.factor(Blood_finaliseds$Oily_fish_intake)

Blood_finaliseds$Processed_meat_intake <- as.character(Blood_finaliseds$Processed_meat_intake)
Blood_finaliseds$Processed_meat_intake[is.na(Blood_finaliseds$Processed_meat_intake)] <- "Unknown"
Blood_finaliseds$Processed_meat_intake <- as.factor(Blood_finaliseds$Processed_meat_intake)

Blood_finaliseds$Length_of_mobile_phone_use <- as.character(Blood_finaliseds$Length_of_mobile_phone_use)
Blood_finaliseds$Length_of_mobile_phone_use[is.na(Blood_finaliseds$Length_of_mobile_phone_use)] <- "Unknown"
Blood_finaliseds$Length_of_mobile_phone_use <- as.factor(Blood_finaliseds$Length_of_mobile_phone_use)

Blood_finaliseds$Processed_meat_intake <- as.character(Blood_finaliseds$Processed_meat_intake)
Blood_finaliseds$Processed_meat_intake[is.na(Blood_finaliseds$Processed_meat_intake)] <- "Unknown"
Blood_finaliseds$Processed_meat_intake <- as.factor(Blood_finaliseds$Processed_meat_intake)

Blood_finaliseds$Nap_during_day <- as.character(Blood_finaliseds$Nap_during_day)
Blood_finaliseds$Nap_during_day[is.na(Blood_finaliseds$Nap_during_day)] <- "Unknown"
Blood_finaliseds$Nap_during_day <- as.factor(Blood_finaliseds$Nap_during_day)

Blood_finaliseds$Ever_addicted_alcohol <- as.character(Blood_finaliseds$Ever_addicted_alcohol)
Blood_finaliseds$Ever_addicted_alcohol[is.na(Blood_finaliseds$Ever_addicted_alcohol)] <- "Unknown/No"
Blood_finaliseds$Ever_addicted_alcohol <- as.factor(Blood_finaliseds$Ever_addicted_alcohol)

Blood_finaliseds$Use_of_sun_uv_protection <- as.character(Blood_finaliseds$Use_of_sun_uv_protection)
Blood_finaliseds$Use_of_sun_uv_protection[is.na(Blood_finaliseds$Use_of_sun_uv_protection)] <- "Unknown"
Blood_finaliseds$Use_of_sun_uv_protection <- as.factor(Blood_finaliseds$Use_of_sun_uv_protection)

Blood_finaliseds$Maternal_smoking_around_birth <- as.character(Blood_finaliseds$Maternal_smoking_around_birth)
Blood_finaliseds$Maternal_smoking_around_birth[is.na(Blood_finaliseds$Maternal_smoking_around_birth)] <- "Unknown"
Blood_finaliseds$Maternal_smoking_around_birth <- as.factor(Blood_finaliseds$Maternal_smoking_around_birth)

Blood_finaliseds$Current_tobacco_smoking <- as.character(Blood_finaliseds$Current_tobacco_smoking)
Blood_finaliseds$Current_tobacco_smoking[is.na(Blood_finaliseds$Current_tobacco_smoking)] <- "Unknown"
Blood_finaliseds$Current_tobacco_smoking <- as.factor(Blood_finaliseds$Current_tobacco_smoking)

Blood_finaliseds$Frequency_of_other_exercises_in_last_4_weeks <- as.character(Blood_finaliseds$Frequency_of_other_exercises_in_last_4_weeks)
Blood_finaliseds$Frequency_of_other_exercises_in_last_4_weeks[is.na(Blood_finaliseds$Frequency_of_other_exercises_in_last_4_weeks)] <- "Unknown"
Blood_finaliseds$Frequency_of_other_exercises_in_last_4_weeks <- as.factor(Blood_finaliseds$Frequency_of_other_exercises_in_last_4_weeks)

Blood_finaliseds$War_exposure <- as.character(Blood_finaliseds$War_exposure)
Blood_finaliseds$War_exposure[is.na(Blood_finaliseds$War_exposure)] <- "Unknown"
Blood_finaliseds$War_exposure <- as.factor(Blood_finaliseds$War_exposure)

Blood_finaliseds$Ever_suicide <- as.character(Blood_finaliseds$Ever_suicide)
Blood_finaliseds$Ever_suicide[is.na(Blood_finaliseds$Ever_suicide)] <- "Unknown"
Blood_finaliseds$Ever_suicide <- as.factor(Blood_finaliseds$Ever_suicide)

Blood_finaliseds$Had_menopause[is.na(Blood_finaliseds$Had_menopause)] <- "No"

Blood_finaliseds$Ethnicity[is.na(Blood_finaliseds$Ethnicity)] <- "Unknown"

Blood_finaliseds$Smoking_status <- as.character(Blood_finaliseds$Smoking_status)
Blood_finaliseds$Smoking_status[is.na(Blood_finaliseds$Smoking_status)] <- "Unknown"
Blood_finaliseds$Smoking_status <- as.factor(Blood_finaliseds$Smoking_status)

Blood_finaliseds$Usual_walking_pace <- as.character(Blood_finaliseds$Usual_walking_pace)
Blood_finaliseds$Usual_walking_pace[is.na(Blood_finaliseds$Usual_walking_pace)] <- "Unknown"
Blood_finaliseds$Usual_walking_pace <- as.factor(Blood_finaliseds$Usual_walking_pace)

Blood_finaliseds$Batch_b001 <- as.factor(Blood_finaliseds$Batch_b001)
(levels(Blood_finaliseds$Batch_b001))
Blood_finaliseds$Medication_cholesterol <- as.factor(Blood_finaliseds$Medication_cholesterol)

colnames(Blood_finaliseds)

Blood_finalised  <- rename(Blood_finaliseds, 
                           WBC = Leukocyte_count,  
                           RBC = Erythrocyte_count ,
                           HT = Haematocrit_percentage , 
                           MCV = Mean_corpuscular_volume ,
                           RDW = Erythrocyte_distribution_width  ,
                           PLT = Platelet_count ,
                           PCT = Platelet_crit ,
                           PDW = Platelet_distribution_width,
                           LY = Lymphocyte_count , 
                           MO = Monocyte_count , 
                           NE = Neutrophill_count , 
                           EO = Eosinophil_count , 
                           RET = Reticulocyte_count , 
                           HLR = High_light_scatter_reticulocyte_count , 
                           HGB = Mean_corpuscular_haemoglobin_conc,
                           CYS = Cystatin_C,
                           PHOS = Phosphate,
                           AST = Aspartate_aminotransferase,
                           HBAIC = Glycated_haemoglobin_HbA1c,
                           APOA = Apolipoprotein_A,
                           APOB = Apolipoprotein_B,
                           HDL = HDL_cholesterol,
                           LDLD = LDL_direct,
                           CHOL = Cholesterol)

colnames(Blood_finalised)

Blood_finalised <- Blood_finalised %>% dplyr::select( -No_cancers, -vaf,
                                                      -YOB, -Thrombocyte_volume,
                                                      -Basophill_count, -Lymphocyte_percentage, -Monocyte_percentage,
                                                      -Neutrophill_percentage, -Eosinophill_percentage, -Basophill_percentage,
                                                      -Nucleated_red_blood_cell_percentage, -Reticulocyte_percentage,
                                                      -Mean_reticulocyte_volume, -Mean_sphered_cell_volume, -Immature_reticulocyte_fraction,
                                                      -High_light_scatter_reticulocyte_percentage,
                                                      -Nap_during_day, -Chest_pain_or_discomfort, -Cough_most_days,
                                                      -Chemical_workplace, -Diesel_exhaust_workplace,
                                                      -Asbestos_workplace, -Paint_glue_thinner_workplace,
                                                      -Pesticide_workplace,-Dusty_workplace,
                                                      -Noisy_workplace, -Able_to_confide, -Miserableness,
                                                      -Frequency_of_depressed_mood_in_last_2_weeks, -Risk_taking, -Facial_ageing,
                                                      -Skin_colour, -Salt_added_to_food, -Lamb_mutton_intake,
                                                      -Pork_intake, -Beef_intake, -Poultry_intake, -Oily_fish_intake, 
                                                      -Length_of_mobile_phone_use, -CYS, -PHOS,
                                                      -AST, -HBAIC, -APOA, -APOB, -LDLD, -HDL, -CHOL,
                                                      -Alcohol_consumed, -Alcohol_intake_yesterday, 
                                                      -Medication_cholesterol, -UK_Biobank_assessment_centre,
                                                      -Usual_walking_pace, -Processed_meat_intake, -Maternal_smoking_around_birth,
                                                      -Use_of_sun_uv_protection, -Frequency_of_other_exercises_in_last_4_weeks,
                                                      -War_exposure, -Ever_suicide, -Ever_addicted_alcohol, -Ethnicity,
                                                      -Mood_swings, -eid.y, -gene)

Blood_finalised <- Blood_finalised[!with(Blood_finalised,is.na(WBC)& is.na(RBC)& is.na(HT)& is.na(MCV)& is.na(RDW)&is.na(PLT)&is.na(PCT)&is.na(PDW)&is.na(LY)& 
                                           is.na(MO)& is.na(NE)& is.na(EO)& is.na(RET)& is.na(HLR)& is.na(HGB)),]

Blood_finalised$Had_menopause <- as.character(Blood_finalised$Had_menopause)
Blood_finalised$Had_menopause[is.na(Blood_finalised$Had_menopause)] <- "Male"
Blood_finalised$Had_menopause <- as.factor(Blood_finalised$Had_menopause)

Blood_finalised$Alcohol_intake_frequency.[is.na(Blood_finalised$Alcohol_intake_frequency.)] <- "Unknown"
Blood_finalised$Alcohol_drink_status[is.na(Blood_finalised$Alcohol_drink_status)] <- "Unknown"
Blood_finalised$Time_since_last_menstrual_period[is.na(Blood_finalised$Time_since_last_menstrual_period)] <- "Unknown"

Blood_finalised$Current_tobacco_smoking[is.na(Blood_finalised$Current_tobacco_smoking)] <- "Unknown"

Blood_finalised$Pack_years_smoking[is.na(Blood_finalised$Pack_years_smoking)] <- 0

setwd("/rds/general/user/iw413/home/Summerproject/Descfriptive_analysis")

continuous_data<- Blood_finalised %>% dplyr::select(age)
continuous_data<- c("age", "Pack_years_smoking", "BMI")

vector<-matrix(NA,nrow=length(continuous_data),ncol=2)
range_valid<-lapply(Blood_finalised %>% dplyr::select(continuous_data[1:length(continuous_data)]),FUN= function(x){
  a <- IQR(x,na.rm=TRUE)*1.5
  vector <- mean(x,na.rm=TRUE)+c(-a,a)})

cts_dataset_cleanup <- Blood_finalised %>% dplyr::select((continuous_data)) 
result <- cts_dataset_cleanup %>% mutate(age = ifelse(age<range_valid$age[1] | age>range_valid$age[2], NA, age))
result <- cts_dataset_cleanup %>% mutate(Pack_years_smoking = ifelse(Pack_years_smoking<range_valid$Pack_years_smoking[1] | Pack_years_smoking>range_valid$Pack_years_smoking[2], NA, Pack_years_smoking))
result <- cts_dataset_cleanup %>% mutate(BMI = ifelse(BMI<range_valid$BMI[1] | BMI>range_valid$BMI[2], NA, BMI))

data <- cbind(result,Blood_finalised %>% dplyr::select(-(continuous_data)))
Blood_finalised <-  data                                                     
Blood_finalised$eid <- as.factor(Blood_finalised$eid)


colnames(Blood_finalised)
saveRDS(Blood_finalised, file = "January_blood_dataset.rds")
