source('lib.R')

data("movie_review")
setDT(movie_review)
setkey(movie_review, id)
set.seed(2016L)
all_ids <- movie_review$id
movie_review$review <- tolower(movie_review$review)
movie_review$review <- as.character(lapply(movie_review$review, cleanString))
train_ids <- sample(all_ids, 4000)
test_ids <- setdiff(all_ids, train_ids)
train <- movie_review[J(train_ids)]
test <- movie_review[J(test_ids)]