##MovieLens Project

##Setting the data sets

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
library(tidyverse)
library(caret)

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

options(timeout = 120)
dl <- "ml-10M100K.zip"
if(!file.exists(dl))
  download.file("https://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)
ratings_file <- "ml-10M100K/ratings.dat"
if(!file.exists(ratings_file))
  unzip(dl, ratings_file)
movies_file <- "ml-10M100K/movies.dat"
if(!file.exists(movies_file))
  unzip(dl, movies_file)
ratings <- as.data.frame(str_split(read_lines(ratings_file), fixed("::"), simplify = TRUE),
                         stringsAsFactors = FALSE)
colnames(ratings) <- c("userId", "movieId", "rating", "timestamp")
ratings <- ratings %>%
  mutate(userId = as.integer(userId),
         movieId = as.integer(movieId),
         rating = as.numeric(rating),
         timestamp = as.integer(timestamp))
movies <- as.data.frame(str_split(read_lines(movies_file), fixed("::"), simplify = TRUE),
                        stringsAsFactors = FALSE)
colnames(movies) <- c("movieId", "title", "genres")
movies <- movies %>%
  mutate(movieId = as.integer(movieId))
movielens <- left_join(ratings, movies, by = "movieId")
# Final hold-out test set will be 10% of MovieLens data
set.seed(1, sample.kind="Rounding") # if using R 3.6 or later
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]
# Make sure userId and movieId in final hold-out test set are also in edx set
final_holdout_test <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")
# Add rows removed from final hold-out test set back into edx set
removed <- anti_join(temp, final_holdout_test)
edx <- rbind(edx, removed)
rm(dl, ratings, movies, test_index, temp, movielens, removed)

##Training the algorithm

library(tidyverse)
library(caret)
library(dslabs)

#Partitioning the edx data set
head(edx)
class(edx)
nrow(edx)
ncol(edx)
names(edx)
edx %>% 
  group_by(movieId) %>% 
  summarize(b_m = mean(rating)) %>%
  ggplot(aes(b_m)) + 
  geom_histogram(color = "black", fill = "green") +
  ggtitle("Histogram of Movie-Specific Effects") +
  theme(plot.title = element_text(hjust=0.5))
edx %>% 
  group_by(userId) %>% 
  summarize(b_u = mean(rating)) %>%
  ggplot(aes(b_u)) + 
  geom_histogram(color = "black", fill = "blue") +
  ggtitle("Histogram of User-Specific Effects") +
  theme(plot.title = element_text(hjust=0.5))
index <- createDataPartition(y = edx$rating, times = 1, p = 0.1, list = FALSE)
edx_train <- edx[-index,] #Training set
head(edx_train)
nrow(edx_train)
ncol(edx_train)
edx_test <- edx[index,] #Test set
edx_test <- edx_test %>%
  semi_join(edx_train, by = "movieId") %>%
  semi_join(edx_train, by = "userId")
head(edx_test)
nrow(edx_test)
ncol(edx_test)

#Accounting for movie effects
mu <- mean(edx_train$rating)
mu
movie_avgs <- edx_train %>%
  group_by(movieId) %>%
  summarize(b_m = mean(rating-mu))
head(movie_avgs, 10)
nrow(movie_avgs)
predicted_ratings <- mu + edx_test %>%
  left_join(movie_avgs, by = 'movieId') %>%
  pull(b_m)
class(predicted_ratings)
length(predicted_ratings)
head(predicted_ratings)
sum(is.na(predicted_ratings))
identical(length(predicted_ratings), nrow(edx_test))
model_rmse <- sqrt(mean((predicted_ratings-edx_test$rating)^2))
model_rmse

#Accounting for movie and user effects
user_avgs <- edx_train %>% 
  left_join(movie_avgs, by='movieId') %>%
  group_by(userId) %>%
  summarize(b_u = mean(rating - mu - b_m))
head(user_avgs, 10)
nrow(user_avgs)
predicted_ratings <- edx_test %>% 
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  mutate(pr = mu + b_m + b_u) %>%
  pull(pr)
class(predicted_ratings)
length(predicted_ratings)
head(predicted_ratings)
identical(length(predicted_ratings), nrow(edx_test))
model_rmse <- sqrt(mean((predicted_ratings-edx_test$rating)^2))
model_rmse

##Calculating the RMSE
mu <- mean(edx$rating) #This time, we use the whole edx data set
mu
#Estimating the movie effects
movie_avgs <- edx %>%
  group_by(movieId) %>%
  summarize(b_m = mean(rating-mu))
head(movie_avgs, 10)
nrow(movie_avgs)
#Estimating the user effects
user_avgs <- edx %>% 
  left_join(movie_avgs, by='movieId') %>%
  group_by(userId) %>%
  summarize(b_u = mean(rating - mu - b_m))
head(user_avgs, 10)
nrow(user_avgs)
#Using the final_holdout_test data set
nrow(final_holdout_test)
predicted_ratings <- final_holdout_test %>% 
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  mutate(pr = mu + b_m + b_u) %>%
  pull(pr)
class(predicted_ratings)
length(predicted_ratings)
head(predicted_ratings)
identical(length(predicted_ratings), nrow(final_holdout_test))
model_rmse <- sqrt(mean((predicted_ratings-final_holdout_test$rating)^2))
model_rmse
