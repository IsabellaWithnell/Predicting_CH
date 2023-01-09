## Predicting gene stratified CH (large clone) with routine blood test information


The scripts are arranged in two parts:


### 1) Data processing scripts and bash files

Access to the UK Biobank dataset is required. 

Run scripts from 1 to 7 to produce a dataset to be used to train the random forest model below.

Steps involve: extracting variables from UK biobank, editing / renaming variables, imputation and matching (cases with gene stratified CH (large clone) are matched based on age and sex to those without CH (large clone)) to produce a final dataset for training below.


### 2) Gene-stratified random forest script

In here is the script used to 1) train the gene-stratified random forest model for ASXL1 CH (large clone) prediction, can be adjusted for each gene, and 2) test the tuned model on the 25% of the dataset which was held out.
