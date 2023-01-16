# Objective: definition of disease status and time to diagnosis from diagnoses
# recorded in Hospital Episode Statistics and death registry data.
install.packages("data.table",repos = "http://cran.us.r-project.org")
library(data.table)
source("functions.R")

# Reading arguments
args <- commandArgs(trailingOnly = TRUE)
def_path <- as.character(args[1])
app_data_path <- as.character(args[2])
hes_main_path <- as.character(args[3])
hes_diag_path <- as.character(args[4])
hes_oper_path <- as.character(args[5])

# Initialising empty lists of codes
icd10_codes <- NULL


# Reading input files
myfiles <- list.files(path = def_path)
if ("diagnoses_icd10_malig.txt" %in% myfiles) {
  icd10_codes <- fread(paste0(def_path, "/diagnoses_icd10_malig.txt"), data.table = FALSE)
  icd10_codes <- toupper(icd10_codes[, 1])
}

# Messages for quality check
print(paste0("Reading ", length(icd10_codes), " ICD10 codes"))



# Loading application data (date of recruitment and eid only)
print("Loading main UK Biobank dataset from:")
print(app_data_path)
column_names <- colnames(fread(app_data_path, data.table = FALSE, nrows = 0))
tmp <- fread(app_data_path, data.table = FALSE, select = which(column_names %in% c("eid", "53-0.0")))
date_recr <- as.Date(tmp[, 2])
names(date_recr) <- tmp[, 1]
rm(tmp)

# Loading HES data
print("Loading main HES dataset from:")
print(hes_main_path)
hes_full <- fread(hes_main_path, data.table = FALSE)
print("Loading HES diagnosis dataset from:")
print(hes_diag_path)
hes <- fread(hes_diag_path, data.table = FALSE)
print("Loading HES operations dataset from:")
print(hes_oper_path)
hes_oper <- fread(hes_oper_path, data.table = FALSE)

# Extracting relevant participants
hes <- hes[hes$eid %in% names(date_recr), ]
hes_oper <- hes_oper[hes_oper$eid %in% names(date_recr), ]
hes_full <- hes_full[hes_full$eid %in% names(date_recr), ]

# Defining episode ID to link the two HES files
hes$episode_id <- paste0(hes$eid, "_", hes$ins_index)
hes_oper$episode_id <- paste0(hes_oper$eid, "_", hes_oper$ins_index)
hes_full$episode_id <- paste0(hes_full$eid, "_", hes_full$ins_index)
rownames(hes_full) <- hes_full$episode_id

# Initialising empty object
mydata <- data.frame(
  eid = names(date_recr),
  date_recr = date_recr,
  date_diagnosis = as.Date(rep(NA, length(date_recr))),
  date_death = as.Date(rep(NA, length(date_recr))),
  case = rep(0, length(date_recr)),
  prevalent_case = rep(0, length(date_recr)),
  incident_case = rep(0, length(date_recr)),
  time_to_diagnosis = rep(NA, length(date_recr))
)
rownames(mydata) <- mydata$eid

head(mydata)

# Re-writing ICD codes (allowing to write e.g. I20-I25 to say I20, I21, I22, I23, I24, I25)
for (type in c("icd10")) {
  input_codes <- eval(parse(text = paste0(type, "_codes")))
  if (length(input_codes) > 0) {
    full_list <- ExpandCodes(input_codes = input_codes)
  } else {
    full_list <- input_codes
  }
  assign(paste0("full_", type), full_list)
}

# Definition of HES columns to use
hes_columns <- c("diag_icd10")
names(hes_columns) <- c("icd10")

# Loop over different codes
unknown_eids <- NULL
for (type in c("icd10")) {
  full_list <- eval(parse(text = paste0("full_", type)))
  if (length(full_list) > 0) {
    print(paste0("Extracting based on ", type, " codes"))
    pb <- txtProgressBar(style = 3)
    for (k in 1:length(full_list)) {
      # Identifying episode IDs of corresponding diagnosis
      if (type %in% c("icd10")) {
        tmp <- as.character(hes$episode_id[grepl(paste0("^", full_list[k]), hes[, hes_columns[type]])])
      } else {
        tmp <- as.character(hes_oper$episode_id[grepl(paste0("^", full_list[k]), hes_oper[, hes_columns[type]])])
      }
      
      if (length(tmp) > 0) {
        # Extracting potential dates
        dates <- hes_full[tmp, c("epistart", "epiend", "admidate", "disdate")]
        
        # Making use of first available date (to avoid missing date as much as possible)
        epistart <- rep(NA, nrow(dates))
        for (i in 1:nrow(dates)) {
          epistart[i] <- min(as.Date(as.character(dates[i, ]), format = "%d/%m/%Y"), na.rm = TRUE)
          # epistart[i] <- min(as.Date(as.numeric(dates[i, ]), origin = "1970-01-01"), na.rm = TRUE)
        }
        epistart <- as.Date(epistart, origin = "1970-01-01")
        
        # Extracting eid of diagnosed participants
        tmp_eid <- as.character(gsub("_.*", "", tmp))
        
        # Extracting the first event by participant
        epistart_missing <- as.numeric(sapply(split(epistart, f = tmp_eid), min))
        epistart_missing[is.infinite(epistart_missing)] <- NA
        epistart <- sapply(split(epistart, f = tmp_eid), min, na.rm = TRUE)
        tmp_eid <- names(epistart)
        epistart <- as.Date(epistart, origin = "1970-01-01")
        
        # Storing eids of participants with missing dates of diagnosis if none is before recruitment (not able to say if prevalent or incident)
        if (sum(is.na(epistart_missing)) > 0) {
          unclear_ids <- tmp_eid[is.na(epistart_missing)]
          unknown_eids <- c(unknown_eids, unclear_ids[(mydata[unclear_ids, "date_recr"] < epistart[unclear_ids])])
        }
        
        # Updating first date of diagnosis if before the current one
        mydata[tmp_eid, "date_diagnosis"] <- as.Date(ifelse((mydata[tmp_eid, "date_diagnosis"] >= epistart) | (is.na(mydata[tmp_eid, "date_diagnosis"])),
                                                            yes = epistart,
                                                            no = mydata[tmp_eid, "date_diagnosis"]
        ),
        origin = "1970-01-01"
        )
        
        # Setting participant as a case
        mydata[tmp_eid, "case"] <- 1
      }
      setTxtProgressBar(pb, k / length(full_list))
    }
    cat("\n")
  }
}

# Definition of prevalent/incident cases based on dates of recruitment and first diagnosis
prevalent_ids <- which(mydata[, "date_diagnosis"] < mydata[, "date_recr"])
mydata[prevalent_ids, "prevalent_case"] <- 1
mydata[prevalent_ids, "incident_case"] <- NA
incident_ids <- which(mydata[, "date_diagnosis"] >= mydata[, "date_recr"])
mydata[incident_ids, "incident_case"] <- 1

head(mydata)

# No classification as incident/prevalent for participants with missing date of diagnosis
mydata[unknown_eids, "prevalent_case"] <- NA
mydata[unknown_eids, "incident_case"] <- NA

# Definition of time to diagnosis from dates of recruitment and first diagnosis
mydata[incident_ids, "time_to_diagnosis"] <- mydata[incident_ids, "date_diagnosis"] - mydata[incident_ids, "date_recr"]

head(mydata)

# Saving outputs

saveRDS(data.frame(mydata), paste0("/rds/general/user/iw413/home/Summerproject/extraction_and_recoding/output_hes_malig.rds"))

