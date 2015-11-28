library("rjson")
library("plyr")
library(ggplot2)
library(maps) # for future mapping
library(lubridate) # for working withe dates and times

#remove all variables and values the environment
rm(list=ls(all=TRUE)) 

#load us map data
all_states <- map_data("state")
#plot all states with ggplot
max_pages=1000 # be respectful, limit the number of pages being polled.
df=data.frame() #stores the consolidated list of issues from SeeClickFix
for (i in 1:max_pages) 
{
  #construct the URL to retrieve a page of data
 url = paste0("http://seeclicktest.com/api/issues.json?at=Burlington,+VT&start=50000&end=0&page=", toString(i), "&num_results=100&sort=issues.created_at")
  # print(url)
  seeclick_data <- fromJSON(paste(readLines(url), collapse=""))
   df1 = ldply (seeclick_data, data.frame, stringsAsFactors = FALSE )

  if ( length(df1)== 0 ) {  #if no more data is available, an empy record is returned. 
    breakFlag = TRUE
    break
  }
  df = rbind(df,df1)        # append the page of data to the overall results. 
    
}
  
# convert updated_raw date/ime into date object
df$date_updated = ymd_hms(df$updated_at_raw)
df$days_since_created = df$minutes_since_created/60/24
#earliest update in data_frame
min(df$date_updated)
# most recent update in data_frame
max(df$date_updated)
#calculate a sequence of months that spans the min and max  dates in the dataframe 
months_spanned = seq(min(df$date_updated),max(df$date_updated), by = 'weeks')

#plot a facete view on status of days since issues was created verus the date the 
#issue was updated. 
qplot(data=df, y= date_updated, x = days_since_created, color = status, breaks = months_spanned) + facet_grid(.~status)


#some simple counts/stats
totalNumIssues= length(df$status)
numClosed= length(df$status[df$status=="Closed"])
numOpen = length(df$status[df$status=="Open"])
numAcknowledged = length(df$status[df$status=="Acknowledged"])
num_reports= length(df)
num_reports


# lets try connecticut
states <- subset(all_states, region %in% c("vermont") )
p <- ggplot()
p <- p + geom_polygon( data=states, aes(x=long, y=lat, group = region),colour="white", fill="grey80" )
p

p <- p + geom_polygon( data=df, aes(x=lng, y=lat, group = minutes_since_created),colour="white", fill="grey80" )
p

p <- p  +  geom_point(data=df, aes(x=df$lng, y=df$lat, size = df$minutes_since_created, alpha=df$minutes_since_created), color="coral1") 
p


p2 = ggplot()

#due to wide range of minutes since created, let's take the log of that factor
df$log_minutes = log(df$minutes_since_created)

p2 <- p2  +  geom_point(data=df, aes(x=df$lng, y=df$lat, size =df$log_minutes, color="coral1")) + geom_text(aes(label = df$minutes_since_created, x=df$lng, y=df$lat), size = 3)
p2
mean(df$minutes_since_created)

p4 = qplot(data = df, x = df$status, y = df$minutes_since_created, color = status ) +xlab("Status of Report") +ylab("Minutes Since Created") 
p4

p5 = qplot(df$minutes_since_created, binwidth=1000) +xlab("Minutes Since Created") +ylab("Number of Reports")
p5


p6 = qplot(data = df, x = status, y = minutes_since_created, color = status ) +xlab("Status of Report") +ylab("Minutes Since Created") + facet_grid(.~rating)
p6

#plot the "open" issues only ->
open_df = subset(df, status=="Open")
p7 = qplot(open_df$minutes_since_created) +xlab("Minutes Since Created") +ylab("Number of Reports")
p7
#plot status faceted by RATING
p8 = qplot(data = open_df, x = status, y = minutes_since_created, color = status, main = " OPEN REPORTS BY RATING" ) +xlab("Status of Report") +ylab("Minutes Since Created") + facet_grid(.~rating)
p8

qplot(data=df, y= updated_at, x = created_at, color = status)

qplot(data=df, y= updated_at_raw, x = minutes_since_created, color = status)
qplot(data=df, y= updated_at_raw, x = minutes_since_created, color = status)

#get rid of duplicates ( about 50 out of 2000 in one case)
x = unique(df)


qplot(data=df, y= updated_at_raw, x = minutes_since_created, color = status) + facet_grid(.~status)


