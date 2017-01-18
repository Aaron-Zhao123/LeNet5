library(ggplot2)
raw_data <- read.table("log/data0105.txt",head=FALSE,sep=",");
names(raw_data) <- c("index","accuracy","entropy");
data_nn <- data.frame(c(1:dim(raw_data)[1]),raw_data["accuracy"],raw_data["entropy"]);
names(data_nn) <- c("index","accuracy","entropy");


ggplot() +
  geom_line(data = data_nn, aes(x = index, y = accuracy), color = "darkred")+
  labs(
    x = "Iterations",
    y = "Training Accuracy"
  )
