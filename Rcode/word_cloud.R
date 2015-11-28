# source / inspiration: http://onertipaday.blogspot.com/2011/07/word-cloud-in-r.html
library(tm)
library(wordcloud)
library(RColorBrewer)

descript_text <- df
descript_text <- descript_text$description[descript_text$description>0]

ds <-VectorSource(descript_text)
descript_text.corpus <- Corpus(ds)
descript_text.corpus <- tm_map(descript_text.corpus, removePunctuation)
descript_text.corpus <- tm_map(descript_text.corpus, tolower)
descript_text.corpus <- tm_map(descript_text.corpus, function(x) removeWords(x, stopwords("english")))
tdm <- TermDocumentMatrix(descript_text.corpus)
m <- as.matrix(tdm)
v <- sort(rowSums(m),decreasing=TRUE)
d <- data.frame(word = names(v),freq=v)
# pal <- brewer.pal(9, "BuGn")
# pal <- pal[-(1:2)]
pal2 <- brewer.pal(8,"Dark2")
# png("wordcloud.png", width=1280,height=800)
png("wordcloud.png", width=3280,height=1800)
wordcloud(d$word,d$freq, scale=c(8,.3),min.freq=2,max.words=100, random.order=T, rot.per=.15, colors=pal2, vfont=c("sans serif","plain"))
dev.off()
