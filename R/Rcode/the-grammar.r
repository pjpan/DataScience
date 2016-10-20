# install.packages("ggplot2")
library(ggplot2)
# http://had.co.nz/ggplot2

# qplot examples -------------------------------------------------------------

qplot(diamonds$cut, diamonds$carat)
qplot(carat, price, data = diamonds)
qplot(carat, price, data = diamonds, colour=clarity)
qplot(carat, price, data = diamonds, geom=c("point", "smooth"), method=lm)

qplot(carat, data = diamonds,
  geom="histogram")
qplot(carat, data = diamonds,
  geom="histogram", binwidth = 1)
qplot(carat, data = diamonds,
  geom="histogram", binwidth = 0.1)
qplot(carat, data = diamonds,
  geom="histogram", binwidth = 0.01)

# aes(x = mpg ^ 2, y = wt / cyl)

# using ggplot() -------------------------------------------------------------
d <- ggplot(diamonds, aes(x=carat, y=price))
d + geom_point()
d + geom_point(aes(colour = carat))
d + geom_point(aes(colour = carat)) + scale_colour_brewer()

ggplot(diamonds) + geom_histogram(aes(x=price))

# Separation of statistcs and geometric elements -----------------------------

p <- ggplot(diamonds, aes(x=price))

p + geom_histogram()
p + stat_bin(geom="area")
p + stat_bin(geom="point")
p + stat_bin(geom="line")

p + geom_histogram(aes(fill = clarity))
p + geom_histogram(aes(y = ..density..))

# Setting vs mapping ---------------------------------------------------------
p <- ggplot(diamonds, aes(x=carat,y=price))

# What will this do?
p + geom_point(aes(colour = "green"))
p + geom_point(colour = "green")
p + geom_point(colour = colour)

