#!/usr/bin/env Rscript

library(ggplot2)
library(ggthemes)
library(plyr)
library(lattice)
library(latticeExtra)
library(gridExtra)
library(grid)
library(latex2exp)

fancy_scientific <- function(l) {
                                        # turn in to character string in scientific notation
    l <- format(l, scientific = TRUE)
                                        # quote the part before the exponent to keep all the digits
    l <- gsub("0e\\+00","0",l)
    l <- gsub("^(.*)e", "'\\1'e", l)
                                        # turn the 'e+' into plotmath format
    l <- gsub("e", "%*%10^", l)
                                        # return this as an expression
    parse(text=l)
}

solver="direct_ginkgo"
cpu_gpu="cuda"
num_procs=18
partition="naive"
comm="onesided"

load_from_files_averaged <- function(my_path, compare){
    myFiles <- list.files(path=my_path, pattern=compare, full.names=F, recursive=FALSE)
    ldf <- vector(mode="list", length=length(myFiles))
    for (k in 1:(length(myFiles)))
    {
        ldf[[k]] <- as.data.frame(read.csv(paste(my_path,myFiles[[k]],sep="") ))
    }
    result <- data.frame(avg = apply(sapply(ldf, function(x) x[, "avg"]), 1, mean))
    result$time_in <- c("boundary_exchange","boundary_update",  "convergence_check", "local_solve", "expand_local_vec" )
    return(result)
}

load_from_files <- function(my_path, compare){
    myFiles <- list.files(path=my_path, pattern=compare, full.names=F, recursive=FALSE)
    myFiles.sorted <- sort(myFiles)
    split <- strsplit(myFiles.sorted, "subd_")
    split <- as.numeric(sapply(split, function(x) x <- sub(".csv", "", x[2])))
    myFiles.correct.order <- myFiles.sorted[order(split)]
    ldf <- vector(mode="list", length=length(myFiles.correct.order))
    ## print(myFiles)
    print(myFiles.correct.order)
    for (k in 1:(length(myFiles.correct.order)))
    {
        ldf[[k]] <- as.data.frame(read.csv(paste(my_path,myFiles.correct.order[[k]],sep="") ))
        ldf[[k]]$process <- k
    }
    return(ldf)
}
num_procs_dir=sprintf("%d",num_procs)
ldf <- load_from_files(sprintf("./%s/%s/%sdomains/%s/%s/512local/",cpu_gpu, comm, num_procs_dir,solver,partition) , "^subd_*")
fulldata2 <- rbind(ldf[[1]])
for (k in 2:num_procs)
{
    fulldata2 <- rbind(fulldata2, ldf[[k]])
}
bar_plot2 <- ggplot(data = fulldata2,
                   mapping =  aes(x = process, y = avg, fill = func))+
    theme_linedraw()+
    geom_bar(stat="identity")+
    scale_x_discrete(limits=fulldata2$process, breaks =c(6,12,18,24,30,36))+
    scale_y_continuous()+
    labs(title = (TeX(sprintf("%s comm, $%d$ subdomains, %s partitioning, Number of rows $=2^{18}$",comm, num_procs, partition))), x = "Subdomain", y = "Time per iteration (s)", color = "Time spent in\n")

ggsave(sprintf("%s_%s_%s_%d_%s_bar.pdf", comm,partition,cpu_gpu,num_procs,solver))


