library(ggplot2)

args <- commandArgs(TRUE)

if (length(args) != 1) {
    cat("Usage: ... <dataFile>\n");
    quit(status=1)
}

data <- read.csv(args[1])
g <- ggplot(data, aes(x= input, y = output, color = func, group = func))
g <- g + geom_line()
ggsave("tmp.png")
