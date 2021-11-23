# Author: Heiletjé van Zyl
# Function: Read in all cleaned vaccine data 
#           Wrangle "created_at" column (i.e., time at which hashtag was used/post was created) into use-able time format 
#           Write the data into time-related dataframes pertaining to these five vaccines and the combination thereof
#------------------------------------------------------------------------------------------------------------------------
# Load required packages for data cleaning purposes
# library(expss)
# library(tidyr)
# library(tidyverse)
# library(dpylr)
# library(lubridate)
#------------------------------------------------------------------------------------------------------------------------
# Function to convert creation time into use-able time formats for plotting purposes (e.g., date and time) and Hawkes process model fitting (e.g., unix time)
vaccine_times <- function(data){
  
  data <- data[, -1] # remove extra, unnecessary column
  
  # create empty vectors to store relevant time information related to vaccines
  unix_time <- c() 
  date_and_time <- c()
  dates <- c()
  
  for (line in 1:nrow(data)){
    split_creation_time <- strsplit(data[line, 1], " ") # split created_at column into strings
    creation_time <- split_creation_time[[1]][c(-1,-5)] # remove unnecessary strings (i.e., day of week and +0000)

    month <- (match(creation_time[1], month.abb)) # match abbreviated month name to be in its numeric form
    date <- paste(creation_time[4],month,creation_time[2], sep = "/") %>% ymd() # extract full date of creation time
    
    date_time <- paste(date, creation_time[3]) # add time to creation date 
    unix <- date_time  %>% ymd_hms  # convert creation time to unix time
    
    date_and_time <- c(date_and_time, date_time)
    unix_time <- c(unix_time, unix)
    
  }
  
  retweet_id <- data$retweet_id # include "retweet_id" column indicating whether tweet is a retweet or not (for Hawkes process model fitting)
  return(cbind(unix_time, date_and_time, dates, retweet_id))
  
}
#------------------------------------------------------------------------------------------------------------------------
# Function to streamline reading in the data 
vaccine_hashtags <- function(vaccine){
  
  setwd(paste0("/Users/heiletjevanzyl/Vaccines/", vaccine)) # change to appropriate directory of user 
  
  months <- c("January", "February", "March", "April", "May", "June", "July")
  
  vaccine_df <- read.csv(paste0(months[1],"_",vaccine,".csv"))
  
  for (d in 2:length(months)){
    df <- read.csv(paste0(months[d],"_",vaccine,".csv"))
    
    vaccine_df <- rbind(vaccine_df, df) # combine all separate month files pertaining to a specific vaccine into one month file
  }
  
  data_frames <- vaccine_times(data = vaccine_df) %>% as.data.frame()

  setwd("/Users/heiletjevanzyl/Vaccines") # change to appropriate directory of user 
  write.csv(data_frames, file = paste0(vaccine, ".csv"))
}
#------------------------------------------------------------------------------------------------------------------------
# Streamline
# vaccines <- c("AZ", "JnJ", "Pfizer", "Moderna", "SputnikV", "allVaccines")
# for (vaccine in vaccines){
#   vaccine_hashtags(vaccine = vaccine)
# }
#------------------------------------------------------------------------------------------------------------------------


