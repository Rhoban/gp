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
    plotPoints <- "type" %in% names(data)
    dst <- sprintf("%s%s.png", getBase(path), getFilePrefix(path))
    observations <- NULL
    if (plotPoints)
    {
        observations <- data[which(data$type == "observation"),]
        data <- data[which(data$type == "prediction"),]
#        print(observations)
    }
    g <- ggplot(data, aes(x= input, y = mean, ymin = min, ymax = max))
    g <- g + geom_line()
#    g <- g + geom_point()
    g <- g + geom_ribbon(alpha = 0.3)
    if (plotPoints)
    {
        print(observations)
        g <- g + geom_point(data=observations,mapping=aes(x=input, y=mean),
                            size=10, color="red", shape = '+')
    }
    ggsave(dst)
}
