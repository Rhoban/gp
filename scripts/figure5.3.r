library(ggplot2)
library(stringr)

getBase <- function(path)
{
    base = str_extract(path,".*/")
    if (is.na(base)) {
        base = "./"
    }
    return(base);
}

getFileName <- function(path)
{
    base <- getBase(path)
    if (base == "./")
    {
        return(path)
    }
    else
    {
        pathLength <- str_length(path)
        baseLength <- str_length(base)
        return(substr(path,baseLength +1,pathLength))
    }
}

getFilePrefix <- function(path)
{
    file <- getFileName(path)
    return(str_extract(file,"[^.]+"))
}

args <- commandArgs(TRUE)

if (length(args) < 1) {
    cat("Usage: ... <dataFiles>\n");
    quit(status=1)
}

for (i in 1:length(args))
{
    path <- args[i]
    data <- read.csv(path)
    data$nbPoints <- as.factor(data$nbPoints)
    plotPoints <- "type" %in% names(data)
    dst <- sprintf("%s%s.png", getBase(path), getFilePrefix(path))
    observations <- NULL
    g <- ggplot(data, aes(x= lengthScale, y = logMarginalLikelihood, group=nbPoints, color=nbPoints))
    g <- g + geom_line()
    g <- g + scale_x_log10()
    g <- g + scale_y_continuous(limits=c(-100,50))
    ggsave(dst)
}
