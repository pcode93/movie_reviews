source('movie_reviews_get_data.R')

result <- lapply(
  list(
    c(5,50),c(5,100),
    c(7,50),c(7,100),
    c(10,100),c(10,150),c(10,200),
    c(12,150),c(12,200),
    c(15,150),c(15,200),c(15,250),c(15,300),c(15,350),c(15,400),
    c(20,300),c(20,350),c(20,400)
  ),
  function(params) {
    print(paste('Skip grams window = ', params[1], ', ', 'Word vector size', params[2]))
    
    wordVectors <- getWordVectors(train$review, params[1], 10L, params[2], 20)
    wvLen <- length(wordVectors[1,])
    wordVectors <- prcomp(wordVectors, scale. = T)$x
    
    train$review <- lapply(
      train$review,
      function(string) wordsToVectors(string, wordVectors)
    )
    
    test$review <- lapply(
      test$review,
      function(string) wordsToVectors(string, wordVectors)
    )
    
    train <- as.data.frame(train)
    train <- train[2:3]
    train[2:(wvLen+1)] <- t(apply(train[2],1,function(x) unlist(x)))
    
    test <- as.data.frame(test)
    test <- test[2:3]
    test[2:(wvLen+1)] <- t(apply(test[2],1,function(x) unlist(x)))
    
    lR <- cv.glmnet(x = as.matrix(train[2:(wvLen+1)]), y = train$sentiment, family = 'binomial', alpha = 0,type.measure = "auc",nfolds = 4,thresh = 1e-3,maxit = 1e3)
    ac1 <- accuracy("LR", lR, as.matrix(test[2:(wvLen+1)]), test$sentiment)
    
    linSVM <- svm(as.matrix(train[2:(wvLen+1)]), y=train$sentiment, type = 'C', kernel = 'linear')
    ac2 <- accuracy("SVM", linSVM, test[2:(wvLen+1)], test$sentiment)
    
    rF <- randomForest(as.factor(sentiment) ~ ., data = train)
    ac3 <- accuracy("RF", rF, test[2:(wvLen+1)], test$sentiment)
    
    nB <- naiveBayes(as.factor(sentiment) ~ ., data = train)
    ac4 <- accuracy("NB", nB, test[2:(wvLen+1)], test$sentiment)
    
    xgb <- xgboost(data = as.matrix(train[2:(wvLen+1)]), label = train$sentiment, nthread = 2, max_depth = 2, nrounds = 200, objective = "binary:logistic", verbose = 0)
    ac5 <- accuracy("XGB", xgb, as.matrix(test[2:(wvLen+1)]), test$sentiment)
    
    list(ac1,ac2,ac3,ac4,ac5)
})

result <- as.data.frame(matrix(unlist(result), nrow=length(unlist(result[1]))))

colnames(result) <- c(
  '(5,50)','(5,100)',
  '(7,50)','(7,100)',
  '(10,100)','(10,150)','(10,200)',
  '(12,150)','(12,200)',
  '(15,150)','(15,200)','(15,250)','(15,300)','(15,350)','(15,400)',
  '(20,300)','(20,350)','(20,400)'
)

rownames(result) <- c(
  'LR', 'SVM', 'RF', 'NB', 'XGB'
)

ngrams <- factor(colnames(result), levels = colnames(result))

par(mar = c(6.5, 6.5, 0.5, 0.5), mgp = c(4, 1, 0))
plot(ngrams, result[1,], axes=F, col="blue", 'l', xlab = 'Skip gram window, vector size', ylab='AUC',ylim=c(0.5,1))
axis(2)
axis(1, at=seq_along(result[1,]),labels=as.character(ngrams), las=2)
lines(ngrams, result[2,], col="red")
lines(ngrams, result[3,], col="green")
lines(ngrams, result[4,], col="black")
lines(ngrams, result[5,], col="purple")
legend('topright', rownames(result), lty=c(1,1), col=c('blue','red','green','black','purple'), ncol=3)

