# =============================================================================
# 1. SETUP & LIBRARY LOADING
# =============================================================================
rm(list = ls())
library(tidyverse)    # loads dplyr, ggplot2, tidyr, forcats, etc.
library(table1)
# conflict_prefer("rename", "dplyr")  # if needed

# =============================================================================
# 2. DATA LOADING
# =============================================================================
# Set working directory for outputs and load primary datasets
setwd("/rds/general/user/iw413/home/Summerproject/outputs")
data      <- readRDS("ukb_recoded.rds")
data_2    <- readRDS("ukb_extracted.rds")
ukb_final <- readRDS("ukb_final.rds")

# Add 'eid' from the row names of data_2
datafinal <- data %>% 
  mutate(eid = rownames(data_2))

# Load malignancy data and merge
setwd("/rds/general/user/iw413/home/Summerproject/extraction_and_recoding")
dff3 <- datafinal
dff1 <- readRDS("output_final_malig.rds") %>% 
  mutate(eid = as.character(eid))
malig_ukbb <- left_join(dff3, dff1, by = "eid")

# =============================================================================
# 3. VARIABLE RECODING & COALESCING
# =============================================================================
# Recode time since last menstrual period using case_when
malig_ukbb <- malig_ukbb %>% 
  mutate(
    Alcohol_intake_yesterday = Alcohol_intake_yesterday.0.0,
    Time_since_last_menstrual_period = case_when(
      Time_since_last_menstrual_period.0.0 %in% c(-1, -3) ~ "Unknown",
      Time_since_last_menstrual_period.0.0 >= 0 & Time_since_last_menstrual_period.0.0 < 3 ~ "Under_3days",
      Time_since_last_menstrual_period.0.0 >= 3 & Time_since_last_menstrual_period.0.0 < 7 ~ "3days_to_1week",
      Time_since_last_menstrual_period.0.0 >= 7 & Time_since_last_menstrual_period.0.0 < 14 ~ "1week_to_2weeks",
      Time_since_last_menstrual_period.0.0 >= 14 & Time_since_last_menstrual_period.0.0 < 21 ~ "2weeks_to_3weeks",
      Time_since_last_menstrual_period.0.0 >= 21 ~ "Over_3weeks",
      is.na(Time_since_last_menstrual_period.0.0) & Sex.0.0 == "Female" ~ "Female_menopause",
      is.na(Time_since_last_menstrual_period.0.0) & Sex.0.0 == "Male" ~ "Male",
      TRUE ~ NA_character_
    ),
    Time_since_last_menstrual_period = as.factor(Time_since_last_menstrual_period)
  )

# Coalesce alcohol-related variables
malig_ukbb <- malig_ukbb %>% 
  mutate(
    Alcohol_consumed = coalesce(Alcohol_consumed.0.0, Alcohol_consumed.1.0,
                                Alcohol_consumed.2.0, Alcohol_consumed.3.0,
                                Alcohol_consumed.4.0),
    Alcohol_intake_yesterday = coalesce(Alcohol_intake_yesterday.0.0,
                                        Alcohol_intake_yesterday.1.0,
                                        Alcohol_intake_yesterday.2.0,
                                        Alcohol_intake_yesterday.3.0,
                                        Alcohol_intake_yesterday.4.0)
  )

# Coalesce workplace exposure variables using a tidy approach
malig_ukbb <- malig_ukbb %>% 
  mutate(
    Chemical_workplace       = coalesce(!!!select(., starts_with("Chemical_workplace.0."))),
    Diesel_exhaust_workplace = coalesce(!!!select(., starts_with("Diesel_exhaust_workplace.0."))),
    Asbestos_workplace       = coalesce(!!!select(., starts_with("Asbestos_workplace.0."))),
    Paint_glue_thinner_workplace = coalesce(!!!select(., starts_with("Paint_glue_thinner_workplace.0."))),
    Pesticide_workplace      = coalesce(!!!select(., starts_with("Pesticide_workplace.0."))),
    Dusty_workplace          = coalesce(!!!select(., starts_with("Dusty_workplace.0.")))
  )

# Drop observations with prevalent blood cancer/disorder
malig_ukbb <- subset(malig_ukbb, prevalent_case != 1)
datafinal  <- malig_ukbb

# =============================================================================
# 4. SELECT BLOOD DATA VARIABLES & CLEAN COLUMN NAMES
# =============================================================================
# Select variables and then drop unwanted ones: "Sexually_molested_child" and "Menstruating_today"
blood_data <- datafinal %>% 
  select(Medication_cholesterol, Alcohol_intake_yesterday, Alcohol_consumed, 
         Alcohol_intake_frequency..0.0, Alcohol_drink_status.0.0, Current_tobacco_smoking.0.0, 
         Time_since_last_menstrual_period, Sex.0.0, YOB.0.0, eid, 
         UK_Biobank_assessment_centre.0.0, Leukocyte_count.0.0, Erythrocyte_count.0.0, 
         Had_menopause.0.0, Haematocrit_percentage.0.0, Mean_corpuscular_volume.0.0,
         Mean_corpuscular_haemoglobin_conc.0.0, Erythrocyte_distribution_width.0.0,
         Platelet_count.0.0, Platelet_crit.0.0, Thrombocyte_volume.0.0, 
         Platelet_distribution_width.0.0, Lymphocyte_count.0.0, Monocyte_count.0.0, 
         Neutrophill_count.0.0, Eosinophil_count.0.0, Basophill_count.0.0, 
         Lymphocyte_percentage.0.0, Monocyte_percentage.0.0, Neutrophill_percentage.0.0, 
         Eosinophill_percentage.0.0, Basophill_percentage.0.0, 
         Nucleated_red_blood_cell_percentage.0.0, Reticulocyte_percentage.0.0,
         Reticulocyte_count.0.0, Mean_reticulocyte_volume.0.0, Mean_sphered_cell_volume.0.0,
         Immature_reticulocyte_fraction.0.0, High_light_scatter_reticulocyte_percentage.0.0, 
         High_light_scatter_reticulocyte_count.0.0, Smoking_status.0.0,
         Cystatin_C_2.0.0, Phosphate_2.0.0, Aspartate_aminotransferase_2.0.0,
         Glycated_haemoglobin_HbA1c_2.0.0, Apolipoprotein_A_2.0.0, Apolipoprotein_B_2.0.0,
         HDL_cholesterol_2.0.0, LDL_direct_2.0.0, Cholesterol_2.0.0, 
         Pack_years_smoking.0.0, BMI.0.0, No_cancers.0.0, Usual_walking_pace.0.0,
         Processed_meat_intake.0.0, Maternal_smoking_around_birth.0.0, 
         Use_of_sun_uv_protection.0.0, Frequency_of_other_exercises_in_last_4_weeks.0.0,
         Menstruating_today.0.0, War_exposure.0.0, Ever_suicide.0.0, 
         Ever_addicted_alcohol.0.0, Ethnicity.0.0, Sexually_molested_child.0.0,
         Length_of_mobile_phone_use.0.0, Nap_during_day.0.0, Oily_fish_intake.0.0,
         Processed_meat_intake.0.0, Poultry_intake.0.0, Beef_intake.0.0, 
         Pork_intake.0.0, Lamb_mutton_intake.0.0, Salt_added_to_food.0.0, 
         Skin_colour.0.0, Facial_ageing.0.0, Mood_swings.0.0, Risk_taking.0.0, 
         Frequency_of_depressed_mood_in_last_2_weeks.0.0, Miserableness.0.0, 
         Able_to_confide.0.0, Chest_pain_or_discomfort.0.0, Noisy_workplace.0.0,
         War_exposure.0.0, Cough_most_days.0.0, Dusty_workplace, 
         Pesticide_workplace, Paint_glue_thinner_workplace,
         Asbestos_workplace, Diesel_exhaust_workplace, Chemical_workplace) %>%
  select(-Sexually_molested_child.0.0, -Menstruating_today.0.0)

# Clean column names by removing extra suffixes
names(blood_data) <- names(blood_data) %>% 
  str_replace_all("_2.0.0", "") %>% 
  str_replace_all(".0.0", "")

# =============================================================================
# 5. MERGE ADDITIONAL DATASETS (GENETIC/CH DATA)
# =============================================================================
# Load FAM files and CH data
setwd("/rds/general/user/iw413/home/Summerproject/outputs")
fam  <- read.table("/rds/general/project/chadeau_ukbb_folder/live/data/project_data/Genetics/ukb22418_c17_b0_v2_s488175.fam", 
                   header = FALSE, stringsAsFactors = FALSE)[-1,]
fam2 <- read.table("ukb22418_c1_b0_v2_s488248.fam", header = TRUE, stringsAsFactors = FALSE)
ch   <- read.table("results_var_ch_all_allgenes_UKB.txt", header = TRUE, stringsAsFactors = FALSE)

fam2 <- fam2 %>% 
  rename(eid = X2039051) %>% 
  mutate(order = as.integer(row.names(.)))

# Merge fam2 with CH data
mergedchfam2 <- merge(fam2, ch, by = "eid", all.x = TRUE)

# Remove duplicate entries (keeping the first occurrence)
mergedchfam22 <- mergedchfam2[!duplicated(mergedchfam2$eid), ]
ch2 <- mergedchfam22 %>% arrange(order)

# Combine 'fam' and ch2 (assuming they align by row order)
dat <- cbind(fam, ch2)
# (Optional: Cross-check using table() if needed)

# Merge blood data with genetic dataset using matching IDs (V1 in fam corresponds to eid)
Blood_finalised <- merge(blood_data, dat, by.x = "eid", by.y = "V1")

# -----------------------------------------------------------------------------
# Drop specific CH columns that are not needed
Blood_finalised <- Blood_finalised %>% 
  select(-c(X2039051.1, X0, X0.1, X1, order, ukb_id, chr, pos, 
            ref_allele, alt_allele, mutect2_filt, consequence, transcript, 
            alt_ad, ref_ad, V2, V3, V4, V5, V6, cds_position, protein_position, 
            amino_acids, codons, c_mut, p_mut, hgvsp, hgvsc, exon, variant_class, 
            impact, existing_variation))
# -----------------------------------------------------------------------------

# =============================================================================
# 6. FINAL DATA PREPARATION & DETAILED RECODING
# =============================================================================
Blood_finalised <- Blood_finalised %>% 
  # Convert variables and compute age
  mutate(
    Sex   = as.factor(Sex),
    symbol = as.factor(symbol),
    age   = 2023 - YOB,
    # Define clone variables based on vaf
    largeclone01 = case_when(
      vaf >= 0.1 ~ "large_clone",
      vaf < 0.1 ~ "small_clone",
      TRUE ~ "no_mutation"
    ),
    largeclone015 = case_when(
      vaf >= 0.15 ~ "large_clone",
      vaf < 0.15 ~ "small_clone",
      TRUE ~ "no_mutation"
    ),
    largeclone02 = case_when(
      vaf >= 0.2 ~ "large_clone",
      vaf < 0.2 ~ "small_clone",
      TRUE ~ "no_mutation"
    ),
    clone = if_else(vaf > 0, "clone", "no_mutation")
  ) %>% 
  filter(!is.na(age))  # remove entries with missing age

# Detailed recoding of categorical variables
Blood_finalised <- Blood_finalised %>% 
  mutate(
    Ethnicity = recode(Ethnicity,
                       "Prefer not to answer" = "Unknown",
                       "Do not know" = "Unknown",
                       "White" = "White/White British",
                       "British" = "White/White British",
                       "Irish" = "White/White British",
                       "Any other white background" = "White/White British",
                       "Mixed" = "Mixed",
                       "White and Black Caribbean" = "Mixed",
                       "White and Black African" = "Mixed",
                       "White and Asian" = "Mixed",
                       "Any other mixed background" = "Mixed",
                       "Asian or Asian British" = "Asian/Asian British",
                       "Indian" = "Asian/Asian British",
                       "Pakistani" = "Asian/Asian British",
                       "Bangladeshi" = "Asian/Asian British",
                       "Any other Asian background" = "Asian/Asian British",
                       "Black or Black British" = "Black/Black British",
                       "Caribbean" = "Black/Black British",
                       "African" = "Black/Black British",
                       "Any other Black background" = "Black/Black British",
                       "Chinese" = "Chinese",
                       "Other ethnic group" = "Other"),
    Had_menopause = recode(Had_menopause,
                           "Prefer not to answer" = "Unknown",
                           "Not sure - had a hysterectomy" = "Unknown_hysterectomy",
                           "Not sure - other reason" = "Unknown"),
    Alcohol_intake_frequency = recode(Alcohol_intake_frequency, "Prefer not to answer" = "Unknown"),
    Smoking_status = recode(Smoking_status, "Prefer not to answer" = "Unknown"),
    Usual_walking_pace = recode(Usual_walking_pace, "Prefer not to answer" = "Unknown"),
    Processed_meat_intake = recode(Processed_meat_intake, 
                                   "Prefer not to answer" = "Unknown", 
                                   "Do not know" = "Unknown", 
                                   "2-4 times a week" = "More_than_2_times_week",
                                   "5-6 times a week" = "More_than_2_times_week",
                                   "Once or more daily" = "More_than_2_times_week"),
    Alcohol_drink_status = recode(Alcohol_drink_status, "Prefer not to answer" = "Unknown"),
    Use_of_sun_uv_protection = recode(Use_of_sun_uv_protection,
                                      "Prefer not to answer" = "Unknown",
                                      "Do not know" = "Unknown"),
    Maternal_smoking_around_birth = recode(Maternal_smoking_around_birth, "Do not know" = "Unknown"),
    Current_tobacco_smoking = recode(Current_tobacco_smoking, "Prefer not to answer" = "Unknown"),
    Frequency_of_other_exercises_in_last_4_weeks = recode(Frequency_of_other_exercises_in_last_4_weeks,
                                                          "Prefer not to answer" = "Unknown",
                                                          "Do not know" = "Unknown"),
    War_exposure = recode(War_exposure,
                          "Prefer not to answer" = "Unknown",
                          "Yes, but not in the last 12 months" = "Yes",
                          "Yes, within the last 12 months" = "Yes"),
    Ever_suicide = recode(Ever_suicide, "Prefer not to answer" = "Unknown"),
    Ever_addicted_alcohol = recode(Ever_addicted_alcohol,
                                   "Prefer not to answer" = "Unknown",
                                   "Do not know" = "Unknown",
                                   "No" = "No"),
    Length_of_mobile_phone_use = recode(Length_of_mobile_phone_use,
                                        "Do not know" = "Unknown",
                                        "Prefer not to answer" = "Unknown"),
    Nap_during_day = recode(Nap_during_day, "Prefer not to answer" = "Unknown"),
    Oily_fish_intake = recode(Oily_fish_intake,
                              "Prefer not to answer" = "Unknown",
                              "Do not know" = "Unknown",
                              "2-4 times a week" = "More_than_2_times_week",
                              "5-6 times a week" = "More_than_2_times_week",
                              "Once or more daily" = "More_than_2_times_week"),
    Poultry_intake = recode(Poultry_intake,
                            "Prefer not to answer" = "Unknown",
                            "Do not know" = "Unknown",
                            "2-4 times a week" = "More_than_2_times_week",
                            "5-6 times a week" = "More_than_2_times_week",
                            "Once or more daily" = "More_than_2_times_week"),
    Beef_intake = recode(Beef_intake,
                         "Prefer not to answer" = "Unknown",
                         "Do not know" = "Unknown",
                         "2-4 times a week" = "More_than_2_times_week",
                         "5-6 times a week" = "More_than_2_times_week",
                         "Once or more daily" = "More_than_2_times_week"),
    Pork_intake = recode(Pork_intake,
                         "Prefer not to answer" = "Unknown",
                         "Do not know" = "Unknown",
                         "2-4 times a week" = "More_than_2_times_week",
                         "5-6 times a week" = "More_than_2_times_week",
                         "Once or more daily" = "More_than_2_times_week"),
    Lamb_mutton_intake = recode(Lamb_mutton_intake,
                                "Prefer not to answer" = "Unknown",
                                "Do not know" = "Unknown",
                                "2-4 times a week" = "More_than_2_times_week",
                                "5-6 times a week" = "More_than_2_times_week",
                                "Once or more daily" = "More_than_2_times_week"),
    Salt_added_to_food = recode(Salt_added_to_food, "Prefer not to answer" = "Unknown"),
    Skin_colour = recode(Skin_colour,
                         "Prefer not to answer" = "Unknown",
                         "Do not know" = "Unknown"),
    Facial_ageing = recode(Facial_ageing,
                           "Prefer not to answer" = "Unknown",
                           "Do not know" = "Unknown"),
    Mood_swings = recode(Mood_swings,
                         "Prefer not to answer" = "Unknown",
                         "Do not know" = "Unknown"),
    Risk_taking = recode(Risk_taking,
                         "Prefer not to answer" = "Unknown",
                         "Do not know" = "Unknown"),
    Frequency_of_depressed_mood_in_last_2_weeks = recode(Frequency_of_depressed_mood_in_last_2_weeks,
                                                         "Prefer not to answer" = "Unknown",
                                                         "Do not know" = "Unknown"),
    Miserableness = recode(Miserableness,
                           "Prefer not to answer" = "Unknown",
                           "Do not know" = "Unknown"),
    Chest_pain_or_discomfort = recode(Chest_pain_or_discomfort,
                                      "Prefer not to answer" = "Unknown",
                                      "Do not know" = "Unknown"),
    Noisy_workplace = recode(Noisy_workplace,
                             "Prefer not to answer" = "Unknown",
                             "Do not know" = "Unknown"),
    Dusty_workplace = recode(Dusty_workplace, "Do not know" = "Unknown"),
    Pesticide_workplace = recode(Pesticide_workplace, "Do not know" = "Unknown"),
    Paint_glue_thinner_workplace = recode(Paint_glue_thinner_workplace, "Do not know" = "Unknown"),
    Asbestos_workplace = recode(Asbestos_workplace, "Do not know" = "Unknown"),
    Diesel_exhaust_workplace = recode(Diesel_exhaust_workplace, "Do not know" = "Unknown"),
    Chemical_workplace = recode(Chemical_workplace, "Do not know" = "Unknown")
  )

# Rename blood measurement variables for clarity
Blood_finalised <- Blood_finalised %>% 
  rename(
    WBC  = Leukocyte_count,
    RBC  = Erythrocyte_count,
    HT   = Haematocrit_percentage,
    MCV  = Mean_corpuscular_volume,
    RDW  = Erythrocyte_distribution_width,
    PLT  = Platelet_count,
    PCT  = Platelet_crit,
    PDW  = Platelet_distribution_width,
    LY   = Lymphocyte_count,
    MO   = Monocyte_count,
    NE   = Neutrophill_count,
    EO   = Eosinophil_count,
    RET  = Reticulocyte_count,
    HLR  = High_light_scatter_reticulocyte_count,
    HGB  = Mean_corpuscular_haemoglobin_conc,
    CYS  = Cystatin_C,
    PHOS = Phosphate,
    AST  = Aspartate_aminotransferase,
    HBAIC= Glycated_haemoglobin_HbA1c,
    APOA = Apolipoprotein_A,
    APOB = Apolipoprotein_B,
    HDL  = HDL_cholesterol,
    LDLD = LDL_direct,
    CHOL = Cholesterol
  )

# Remove variables no longer needed from the blood dataset
Blood_finalised <- Blood_finalised %>% 
  select(-c(No_cancers, vaf, YOB, Thrombocyte_volume,
            Basophill_count, Lymphocyte_percentage, Monocyte_percentage,
            Neutrophill_percentage, Eosinophill_percentage, Basophill_percentage,
            Nucleated_red_blood_cell_percentage, Reticulocyte_percentage,
            Mean_reticulocyte_volume, Mean_sphered_cell_volume, 
            Immature_reticulocyte_fraction, High_light_scatter_reticulocyte_percentage,
            Nap_during_day, Chest_pain_or_discomfort, Cough_most_days,
            Chemical_workplace, Diesel_exhaust_workplace, Asbestos_workplace,
            Paint_glue_thinner_workplace, Pesticide_workplace, Dusty_workplace,
            Noisy_workplace, Able_to_confide, Miserableness,
            Frequency_of_depressed_mood_in_last_2_weeks, Risk_taking, Facial_ageing,
            Skin_colour, Salt_added_to_food, Lamb_mutton_intake, Pork_intake,
            Beef_intake, Poultry_intake, Oily_fish_intake, Length_of_mobile_phone_use,
            CYS, PHOS, AST, HBAIC, APOA, APOB, LDLD, HDL, CHOL,
            Alcohol_consumed, Alcohol_intake_yesterday, Medication_cholesterol,
            UK_Biobank_assessment_centre, Usual_walking_pace, Processed_meat_intake,
            Maternal_smoking_around_birth, Use_of_sun_uv_protection,
            Frequency_of_other_exercises_in_last_4_weeks, War_exposure,
            Ever_suicide, Ever_addicted_alcohol, Ethnicity, Mood_swings, eid.y, gene))

# Handle missing values for key variables and set defaults
Blood_finalised <- Blood_finalised %>% 
  filter(!(is.na(WBC) & is.na(RBC) & is.na(HT) & is.na(MCV) & is.na(RDW) &
           is.na(PLT) & is.na(PCT) & is.na(PDW) & is.na(LY) & is.na(MO) &
           is.na(NE) & is.na(EO) & is.na(RET) & is.na(HLR) & is.na(HGB))) %>%
  mutate(
    Had_menopause = if_else(is.na(Had_menopause), "Male", Had_menopause),
    Alcohol_intake_frequency = replace_na(Alcohol_intake_frequency, "Unknown"),
    Alcohol_drink_status = replace_na(Alcohol_drink_status, "Unknown"),
    Time_since_last_menstrual_period = replace_na(Time_since_last_menstrual_period, "Unknown"),
    Current_tobacco_smoking = replace_na(Current_tobacco_smoking, "Unknown"),
    Pack_years_smoking = replace_na(Pack_years_smoking, 0)
  )

Blood_finalised$eid <- as.factor(Blood_finalised$eid)

# =============================================================================
# 7. CONTINUOUS DATA OUTLIER CLEANING
# =============================================================================
setwd("/rds/general/user/iw413/home/Summerproject/Descfriptive_analysis")

# Define continuous variables to check
continuous_vars <- c("age", "Pack_years_smoking", "BMI")

# Compute IQR-based ranges for each continuous variable
range_valid <- lapply(Blood_finalised %>% select(all_of(continuous_vars)), function(x) {
  a <- IQR(x, na.rm = TRUE) * 1.5
  mean_x <- mean(x, na.rm = TRUE)
  c(lower = mean_x - a, upper = mean_x + a)
})

# Clean outliers by replacing values outside the range with NA
cts_dataset <- Blood_finalised %>% select(all_of(continuous_vars))
cts_dataset <- cts_dataset %>%
  mutate(
    age = ifelse(age < range_valid$age["lower"] | age > range_valid$age["upper"], NA, age),
    Pack_years_smoking = ifelse(Pack_years_smoking < range_valid$Pack_years_smoking["lower"] | 
                                  Pack_years_smoking > range_valid$Pack_years_smoking["upper"], NA, Pack_years_smoking),
    BMI = ifelse(BMI < range_valid$BMI["lower"] | BMI > range_valid$BMI["upper"], NA, BMI)
  )

# Replace the continuous columns in the main dataset with the cleaned ones
Blood_finalised <- bind_cols(cts_dataset, Blood_finalised %>% select(-all_of(continuous_vars)))
Blood_finalised$eid <- as.factor(Blood_finalised$eid)

# =============================================================================
# 8. SAVE FINAL DATASET
# =============================================================================
setwd("/rds/general/user/iw413/home/Summerproject/Dataprep")
saveRDS(Blood_finalised, file = "January_blood_dataset_no_multiple_CH.rds")
# For datasets including individuals with multiple clones, use:
# saveRDS(Blood_finalised, file = "January_blood_dataset_with_multiple_CH.rds")

getwd()

