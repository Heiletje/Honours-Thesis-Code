# Authors: Heiletjé van Zyl and Joshua Giese 
# Function: Read in and clean the full hydrated COVID-19 Vaccine Twitter data
#           Remove the unnecessary columns and rows
#           Extract hashtags relating to five specific COVID-19 vaccines
#           Write the data into cleaned dataframes pertaining to these five vaccines and the combination thereof
#------------------------------------------------------------------------------------------------------------------------
# Load required packages for data cleaning purposes
# library(expss)
# library(tidyr)
# library(tidyverse)
# library(dpylr)
#------------------------------------------------------------------------------------------------------------------------
# Set up dictionaries containing relevant vaccine hashtags 
moderna_dictionary <- c("moderna", "modernagang", "modernahammeredhank", "modernamafia", 
                        "modernamurderers", "modernavaccine", "modernavaccinesday", 
                        "teammoderna", "thankmoderna")

JnJ_dictionary <- c("johnsonandjohnshon", "johnsonandjohnshonvaccine", "johnsonjohnson", 
                    "johnsonvariant", "johnsonvariant2", "johnsonvariantday", 
                    "johnsonvarient", "thejohnsonvariant", "jnj")

pfizer_dictionary <- c("pfizer", "pfizerbiontech", "pfizercovidvaccine", "biontechpfizer",
                       "pfizercovid19vaccine", "pfizerjab", "pfizervaccine",
                       "pfizercrimeagainsthumanity", "myocarditisafterpfizer", 
                       "pfizereyewash", "pfizerforall", "pfizerforindiankids", 
                       "pfizergang", "pfizergirl", "pfizergirlsummer", "pfizerleak", 
                       "pfizerlobby", "pfizermisermorrison", "pfizeroverseaspakistanis", 
                       "pfizerpandemicprofiteering", "pfizerpoison", "pfizerprincess", 
                       "pfizerproud", "pfizerterror", "pfizerunethicalpractices", 
                       "pfizervaccinesideeffects", "teampfizer", "thankpfizer", "biontech", 
                       "teambiontech", "biontechleaks")

AZ_dictionary <- c("astravzeneca", "astrazenaca", "astrazeneca", "astrazenecacontract", 
                   "astrazenecacovishield", "astrazenecavaccine", "astrazenecawatch", 
                   "astrazenek", "astrazenica", "astrazenicavaccine", "oxfordastrazeneca",
                   "oxfordastrazenecaheroes", "azvaccine", "oxfordaz", "az")


sputnikv_dictionary <- c("agpcombatingcovid19_sputnikv", "agpsputnikpakistan", 
                        "bringbacksputnik", "sputinikv", "sputinikv", "sputnik", 
                        "sputnikbreaking", "sputniklight", "sputnikupdates", 
                        "sputnikv", "sputnikv4victory", "sputnikvaccinated", 
                        "sputnikvaccine", "sputnikvaccineinkenya", "sputnikvaccineke",
                        "sputnikvaccinekenya", "sputnikvaccinelifesaver", 
                        "sputnikvaccineregistration", "sputnikvargentina", 
                        "sputnikvgarantizavida", "sputnikvparalavida", 
                        "welcomesputnikpakistan", "wewantsputnik", "wewantsputnikv")

vaccine_dictionary <- list(moderna_dictionary, JnJ_dictionary, pfizer_dictionary,
                           AZ_dictionary, sputnikv_dictionary, c(moderna_dictionary, JnJ_dictionary, pfizer_dictionary,
                                                                AZ_dictionary, sputnikv_dictionary))

dictionary_order <- c("Moderna", "JnJ", "Pfizer", "AZ", "SputnikV", "allVaccines")
#------------------------------------------------------------------------------------------------------------------------
# Function to extract rows of data with hashtags that mention the vaccines of interest
get_hashtags <- function(data){
  
  vaccine_dataframes <- list()
  
  storage <- c()  # stores all indices
  
  for(i in 1:nrow(data)){
    temp <- tolower(data$hashtags[[i]]) # alters hashtags to be in lowercase form
    for(j in 1:6){
      checker <- temp %in% vaccine_dictionary[[j]] # checks in which dictionary the vaccine hashtag exists
      if (sum(checker) >= 1){ # if the vaccine hashtag exists in a dictionary, track index (row (i) and dictionary (j))
        storage <- rbind(storage, c(i,j)) # creates tuple of the row index and dictionary (i.e., related to type of vaccine) 
      }
    }
  }
  for (j in 1:6){
    dictionary_index <- which(storage[,2] == j)
    vaccine_dataframes[[j]] <- as.data.frame(data[storage[dictionary_index,1],]) 
  }
  return(vaccine_dataframes)
}
#------------------------------------------------------------------------------------------------------------------------
# Function to streamline reading in the data and extracting relevant tweets
clean_hashtags <- function(month){
  
  setwd(paste0("/Users/heiletjevanzyl/", month))
  all_dates <- list.files(pattern = "*.csv") # find all files in directory (each date in the month)
  
  # read in the appropriate columns and remove unnecessary rows in each date file of a month 
  master_df <- fread(all_dates[1], select = c("created_at", "hashtags", "id", "retweet_id",
                                              "retweet_count", "favorite_count", 
                                              "user_followers_count", "user_verified")) %>% filter(hashtags != "") 
  for (d in 2:length(all_dates)){
    month_df <- fread(all_dates[d], select = c("created_at", "hashtags", "id", "retweet_id",
                                               "retweet_count", "favorite_count", 
                                               "user_followers_count", "user_verified")) %>% filter(hashtags != "")
    
    master_df <- rbind(master_df, month_df) # combine all separate date files into one month file
  }
  data_frames <- get_hashtags(data = master_df)
  setwd("/Users/heiletjevanzyl/All Months Data") # change to appropriate directory of user 
  for (csv_files in 1:6){ # writes csv files for each of the five vaccines and an overall csv file (for all five vaccines combined)
    write.csv(data_frames[[csv_files]], file = paste0(month, "_", dictionary_order[csv_files],".csv"))
  }
}
#------------------------------------------------------------------------------------------------------------------------
# # Streamline
# months <- c("January", "February", "March", "April", "May", "June", "July", "August")
# for (month in months){
#   clean_hashtags(month = month)
# }
#------------------------------------------------------------------------------------------------------------------------


