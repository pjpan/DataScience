library(data.table)  # faster fread() and better weekdays()
library(dplyr)       # consistent data.frame operations
library(purrr)       # consistent & safe list/vector munging
library(tidyr)       # consistent data.frame cleaning
library(lubridate)   # date manipulation
library(countrycode) # turn country codes into pretty names
library(ggplot2)     # base plots are for Coursera professors
library(scales)      # pairs nicely with ggplot2 for plot label formatting
library(gridExtra)   # a helper for arranging individual ggplot objects
library(ggthemes)    # has a clean theme for ggplot2
library(viridis)     # best. color. palette. evar.
library(DT)          # prettier data.frame output
library(svglite)

# oad data
setwd(getwd())

attacks <- tbl_df(fread(input = "D:/gitcode/Practice/Rplot/Heatmaps/eventlog.csv"))


load("D:/gitcode/Practice/Rplot/Heatmaps/attack.RData")
# 转换时刻和周数
make_hr_wkday <- function(cc, ts, tz) {
  
  real_times <- ymd_hms(ts, tz=tz[1], quiet=TRUE)
  
  data_frame(source_country=cc,
             wkday=weekdays(as.Date(real_times, tz=tz[1])),
             hour=format(real_times, "%H", tz=tz[1]))
  
}

group_by(attacks, tz) %>%
  do(make_hr_wkday(.$source_country, .$timestamp, .$tz)) %>% 
  ungroup() %>% 
  mutate(wkday=factor(wkday,
                      levels=levels(weekdays(0, FALSE)))) -> attacks


datatable(head(attacks))

wkdays <- count(attacks, wkday, hour)

datatable(head(wkdays))

gg <- ggplot(wkdays, aes(x=hour, y=wkday, fill=n))
gg <- gg + geom_tile(color="white", size=0.1)
gg <- gg + scale_fill_viridis(name="# Events", label=comma)
gg <- gg + coord_equal()
gg <- gg + labs(x=NULL, y=NULL, title="Events per weekday & time of day")
gg <- gg + theme_tufte(base_family="Helvetica")
gg <- gg + theme(plot.title=element_text(hjust=0))
gg <- gg + theme(axis.ticks=element_blank())
gg <- gg + theme(axis.text=element_text(size=7))
gg <- gg + theme(legend.title=element_text(size=8))
gg <- gg + theme(legend.text=element_text(size=6))
gg

#  facet with country
count(attacks, source_country) %>% 
  mutate(percent=percent(n/sum(n)), count=comma(n)) %>% 
  mutate(country=sprintf("%s (%s)",
                         countrycode(source_country, "iso2c", "country.name"),
                         source_country)) %>% 
  arrange(desc(n)) -> events_by_country

datatable(events_by_country[,5:3])

filter(attacks, source_country %in% events_by_country$source_country[3:12]) %>% 
  count(source_country, wkday, hour) %>% 
  ungroup() %>% 
  left_join(events_by_country[,c(1,5)]) %>% 
  complete(country, wkday, hour, fill=list(n=0)) %>% 
  mutate(country=factor(country,
                        levels=events_by_country$country[3:12])) -> cc_heat


gg <- ggplot(cc_heat, aes(x=hour, y=wkday, fill=n))
gg <- gg + geom_tile(color="white", size=0.1)
gg <- gg + scale_fill_viridis(name="# Events")
gg <- gg + coord_equal()
gg <- gg + facet_wrap(~country, ncol=2)
gg <- gg + labs(x=NULL, y=NULL, title="Events per weekday & time of day by country\n")
gg <- gg + theme_tufte(base_family="Helvetica")
gg <- gg + theme(axis.ticks=element_blank())
gg <- gg + theme(axis.text=element_text(size=5))
gg <- gg + theme(panel.border=element_blank())
gg <- gg + theme(plot.title=element_text(hjust=0))
gg <- gg + theme(strip.text=element_text(hjust=0))
gg <- gg + theme(panel.margin.x=unit(0.5, "cm"))
gg <- gg + theme(panel.margin.y=unit(0.5, "cm"))
gg <- gg + theme(legend.title=element_text(size=6))
gg <- gg + theme(legend.title.align=1)
gg <- gg + theme(legend.text=element_text(size=6))
gg <- gg + theme(legend.position="bottom")
gg <- gg + theme(legend.key.size=unit(0.2, "cm"))
gg <- gg + theme(legend.key.width=unit(1, "cm"))
gg



count(attacks, source_country, wkday, hour) %>% 
  ungroup() %>% 
  left_join(events_by_country[,c(1,5)]) %>% 
  complete(country, wkday, hour, fill=list(n=0)) %>% 
  mutate(country=factor(country,
                        levels=events_by_country$country)) -> cc_heat2

# To get individual scales for each country we need to make n separate ggplot object and combine then using gridExtra::grid.arrange. 
lapply(events_by_country$country[1:16], function(cc) {
  gg <- ggplot(filter(cc_heat2, country==cc), 
               aes(x=hour, y=wkday, fill=n, frame=country))
  gg <- gg + geom_tile(color="white", size=0.1)
  gg <- gg + scale_x_discrete(expand=c(0,0))
  gg <- gg + scale_y_discrete(expand=c(0,0))
  gg <- gg + scale_fill_viridis(name="")
  gg <- gg + coord_equal()
  gg <- gg + labs(x=NULL, y=NULL, 
                  title=sprintf("%s", cc))
  gg <- gg + theme_tufte(base_family="Helvetica")
  gg <- gg + theme(axis.ticks=element_blank())
  gg <- gg + theme(axis.text=element_text(size=5))
  gg <- gg + theme(panel.border=element_blank())
  gg <- gg + theme(plot.title=element_text(hjust=0, size=6))
  gg <- gg + theme(panel.margin.x=unit(0.5, "cm"))
  gg <- gg + theme(panel.margin.y=unit(0.5, "cm"))
  gg <- gg + theme(legend.title=element_text(size=6))
  gg <- gg + theme(legend.title.align=1)
  gg <- gg + theme(legend.text=element_text(size=6))
  gg <- gg + theme(legend.position="bottom")
  gg <- gg + theme(legend.key.size=unit(0.2, "cm"))
  gg <- gg + theme(legend.key.width=unit(1, "cm"))
  gg
}) -> cclist

cclist[["ncol"]] <- 2

do.call(grid.arrange, cclist)




