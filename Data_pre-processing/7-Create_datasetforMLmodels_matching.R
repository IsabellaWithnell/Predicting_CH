library(Matching)
library(dplyr)

setwd("~/Summerproject/Descfriptive_analysis")
impu <- load(file='imputation.Rda')
impucov <- load(file='imputationcovariates.Rda')
noimpu <- readRDS('noimputationnew.rds')

data <- imputednew[[1]]
data <- as.data.frame(data)
covdata <- imputedcov[[1]]
covdata <- as.data.frame(covdata)

datap <- merge(covdata, data, by.x="eid", by.y="eid")
datap <- merge(datap, noimpu, by.x="eid", by.y="eid")


datap<-subset(datap, clone!="small")
datap$clone <- droplevels(datap$clone)


levels(datap$symbol)

#Nchange the symbol here to the gene you want
datap<-subset(datap, symbol=="No_mutation" | symbol=="JAK2")

datap$symbol <- droplevels(datap$symbol)

datap$symbol <- factor(datap$symbol, 
                       levels=c("JAK2", "No_mutation"), 
                       labels=c("TRUE", "FALSE"))


datap$symbol=ifelse(datap$symbol=="TRUE",1,0)

levels(datap$symbol) <- c(TRUE,FALSE)

datap$symbol <- as.logical(datap$symbol)

table(datap$symbol )
Tr <- datap$symbol
eid <- datap$eid

covariates <- datap %>% dplyr::select(sex, age, batch)

X <- data.matrix(covariates)
rr <- Match(Y=eid, Tr=Tr, X=X, M=1, ties=FALSE)

new_list <- rr[[6]] 
str(new_list)
YDF <- new_list[[1]] 
str(YDF)


EID <- new_list[[1]] 
EID <- as.data.frame(EID) 

NUM <- new_list[[3]] 
str(NUM)
typeof(NUM)

nnum <- as.data.frame(NUM) 

together <- cbind(nnum, EID)

saveRDS(together, file="together.rds")

blended <- together

# Merge with datap 
library(dplyr)
library(MASS)

Merged <- merge(blended, datap, by.x="EID", by.y="eid")

saveRDS(Merged, file="geneJAK2OCT.rds")

#Cleaning

setwd("~/Summerproject/Descfriptive_analysis")

merge <- readRDS("geneJAK2OCT.rds")

Mergednew <- merge %>% dplyr::select(-EID, -Had_menopause, -sex.x, -age.x, -batch.x)

colnames(Mergednew) = gsub(pattern = "y*", replacement = "", x = colnames(Mergednew))

colnames(Mergednew) = gsub(pattern = "\\.*", replacement = "", x = colnames(Mergednew))

datasetformodels <- Mergednew

setwd("~/Summerproject/outputs")


datasetformodels <- rename(datasetformodels, Symbol = smbol)
datasetformodels <- rename(datasetformodels, Sex = sex)
datasetformodels <- rename(datasetformodels, Age = age)
datasetformodels <- rename(datasetformodels, Batch = batch)
datasetformodels <- rename(datasetformodels, Pack_years_smoking = Pack_ears_smoking)
datasetformodels <- rename(datasetformodels, Alcohol_intake_frequency = Alcohol_intake_frequenc)

datasetformodels <- datasetformodels %>% dplyr::select( -CH, -clone)

datasetformodels$Alcohol_intake_frequency<- as.factor(datasetformodels$Alcohol_intake_frequency)

datasetformodels$Symbol<- factor(datasetformodels$Symbol, labels=c("No", "Yes"))

datasetformodels$Batch <- as.numeric(datasetformodels$Batch) 

datasetformodels$Age <- as.numeric(datasetformodels$Age) 

xdata = datasetformodels %>%  dplyr::select(-BMI, -Symbol)
xall = datasetformodels %>% dplyr::select(BMI, Symbol)
data= cbind(xdata, xall)

saveRDS(data, file = "geneJAK2OCT.rds")

sum(is.na(data))
write.csv(data,"~/Summerproject/outputs/python_geneJAK2oct.csv", row.names = FALSE)






