library(tidyverse)
library(readr)
library(caret)
library(mlbench)
library(ranger)
library(rebus)
library(assertive)

#Import data 
coffee <- read_csv("https://raw.githubusercontent.com/Nikiboy26/Coffee_quality_ML/main/Coffee%20Dataset.csv") %>%
  rename_all(tolower) #I like when there's consistency with names, makes analysis easier

View(coffee) #Take a look at the dataset 

#Rename first column
coffee<-coffee%>%
  rename("id" = "x1")

#See column names 
colnames(coffee)

#Select important columns for future predictions 
coffee_sub <- coffee%>%
  select(id, owner, country.of.origin, farm.name, mill, company, region, harvest.year, 
         producer, variety, processing.method, category.one.defects, 
         quakers, category.two.defects, altitude_mean_meters, total.cup.points)

#Let's see 10 most popular countries of origin
head(coffee_sub%>%
  count(country.of.origin)%>%
  arrange(desc(n)), 15)%>%
  ggplot(aes(x = reorder(country.of.origin,-n), y = n)) + geom_bar(stat = "identity") + 
  labs(x = "Country", y = "Coffee")

#Now rare countries
head(coffee_sub%>%
       count(country.of.origin)%>%
       arrange(n), 15)%>%
  ggplot(aes(x = reorder(country.of.origin,-n), y = n)) + geom_bar(stat = "identity") + 
  labs(x = "Country", y = "Coffee")

#Remove countries with less than 6 bags 
countries <- head(coffee_sub%>% 
                    count(country.of.origin)%>%
                    arrange(n), 14) #Create a vector with 14 rarest countries in dataset

coffee_sub <- coffee_sub%>%
  filter(!country.of.origin %in% c(countries$country.of.origin)) #Remove these countries

#See countries with lowest amount of coffee now 
head(coffee_sub%>%
       count(country.of.origin)%>%
       arrange(n), 10)%>%
  ggplot(aes(x = reorder(country.of.origin,-n), y = n)) + geom_bar(stat = "identity") + labs(x = "Country", y = "Coffee")


#Get distinct values 
coffee_sub%>% 
  summarise_all(n_distinct)

#Subset again, remove columns with too many distinct values, so we can run model later
coffee_sub <- coffee_sub%>%
  select(-c(owner, farm.name, mill, region, producer, company))

#CLEANING 
#YEARS 
unique(coffee_sub$harvest.year) #Get unique year values

coffee_sub$harvest.year <- str_replace_all(coffee_sub$harvest.year, "08/09 crop", "2008")
coffee_sub$harvest.year <- str_replace_all(coffee_sub$harvest.year, "4T/10", "2010")

#Get first year in two years formar
coffee_sub$harvest.year <- as.numeric(str_extract(coffee_sub$harvest.year, pattern = "20[0-1][0-9]"))

#Check new 'year' column
coffee_sub%>%
  count(harvest.year)

  
#GEt rid of NA's 
coffee_sub <- coffee_sub%>% na.omit()

length(which(is.na(coffee_sub))) #Make sure no NA's

#OUT OF RANGE VALUES
ggplot(coffee_sub, aes(x = altitude_mean_meters, y =  total.cup.points, color = processing.method)) + 
  geom_point()

#Take a look at coffee that grows too high
coffee%>%
  filter(altitude_mean_meters > 5000)%>%
  select(country.of.origin, harvest.year, altitude_mean_meters, altitude, region)
library(assertive)
assert_all_are_in_closed_range(coffee_sub$altitude_mean_meters, lower = 0, upper = 5000) #Check range

#Replace some values
coffee_sub$altitude_mean_meters <- str_replace_all(coffee_sub$altitude_mean_meters, "110000", "1100")
coffee_sub$altitude_mean_meters <- str_replace_all(coffee_sub$altitude_mean_meters, "11000", "1100")
coffee_sub$altitude_mean_meters <- str_replace_all(coffee_sub$altitude_mean_meters, "190164", "1901")


### LET'S PREDICT 
coffee_sub <- coffee_sub%>%
  select(-id)

set.seed(150)
# Shuffle row indices: rows
rows <- sample(nrow(coffee_sub))

# Randomly order data
shuffled_coffee <- coffee_sub[rows,]

# Determine row to split on: split
split <- round(nrow(coffee_sub) * .80)

# Create train
train <- shuffled_coffee[1:split,]

# Create test
test <- shuffled_coffee[(split +1):nrow(shuffled_coffee),]

#Before I run the models, I decided to use RMSE to evaluate how good models are
#The lesser the better. But how small? I want it to be smaller than Standart Deviation 
#Get SD to see how good our RMSE is 
sd <- sd(coffee_sub$total.cup.points)

#Cross valiadtion model 
set.seed(66)
# Fit lm model using 10-fold CV: model
model_cv <- train(
  total.cup.points ~., 
  coffee_sub,
  method = "lm",
  trControl = trainControl(
    method = "cv", 
    number = 10,
    verboseIter = TRUE
  )
)

p <- predict(model_cv, test)

# Compute errors: error
error <- p - test[,"total.cup.points"]

# Calculate RMSE
rmse_cv <-  sqrt(mean(error$total.cup.points^2)) #SD = 2.610294
rmse_cv


#RANDOM FOREST 
set.seed(60)
model_rf <- train(
  total.cup.points ~.,
  tuneLength = 1,
  data = coffee_sub, 
  method = "ranger",
  trControl = trainControl(
    method = "cv", 
    number = 5, 
    verboseIter = TRUE
  )
)

p <- predict(model_rf, test)

# Compute errors: error
error <- p - test[,"total.cup.points"]

# Calculate RMSE
rmse_rf <-  sqrt(mean(error$total.cup.points^2)) #SD = 2.610294
rmse_rf

assert_all_are_in_closed_range(error$total.cup.points, lower = -2, upper = 2)

#Get SD to see how good our RMSE is 
sd 