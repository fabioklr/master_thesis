twitter_filenames <- list.files(pattern = 'tweets_unternehmen[1-8].RData')
lapply(twitter_filenames, load, .GlobalEnv)

tweets_unternehmen <- ls(pattern = "tweets_unternehmen[1-8]")
company_tweets <- data.frame()
for (i in 1:8) {
  print(paste(i, "start"))
  d <- get(tweets_unternehmen[i])[c('author_id', 'created_at', 'id', 'lang', 'referenced_tweets', 'text')]
  for (j in 1:nrow(d)) {
    if (is.null(d$referenced_tweets[[j]][1, 1])) {
      d$reference[j] <- "keep"
    }
    else if (d$referenced_tweets[[j]][1, 1] == "quoted" |
             d$referenced_tweets[[j]][1, 1] == "retweeted") {
      d$reference[j] <- "keep"
    }
    else if (d$referenced_tweets[[j]][1, 1] == "replied_to") {
      d$reference[j] <- "discard"
    }
    print(j)
  }
  d <- d[d$reference == "keep",]
  d[, c('referenced_tweets', 'reference')] <- list(NULL)
  company_tweets <- rbind(company_tweets, d)
  print(paste(i, "end")))
}

# Test code with random sample toy data
toy <- data.frame()
for (i in 1:8) {
  d <- get(tweets_unternehmen[i])[c('author_id', 'created_at', 'id', 'lang', 'referenced_tweets', 'text')]
  toy <- rbind(toy, d[sample(1:nrow(d), 100),])
}

for (j in 1:nrow(toy)) {
  if (is.null(toy$referenced_tweets[[j]][1, 1])) {
    toy$reference[j] <- "keep"
  }
  else if (toy$referenced_tweets[[j]][1, 1] == "quoted") {
    toy$reference[j] <- "keep"
  }
  else if (toy$referenced_tweets[[j]][1, 1] == "retweeted") {
    toy$reference[j] <- "keep"
  } 
  else {
    toy$reference[j] <- "discard"
  }
}

toy <- toy[toy$reference == "keep",]
toy <- subset(toy, select = -reference)
save(toy, file = "toy.RData")
