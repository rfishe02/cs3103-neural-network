
sevenTest <- read.csv(file.choose())

sevenTrain <- read.csv(file.choose())

sevenWeight <- read.csv(file.choose())

#########################
# Summary of the data
############

summary(sevenTest)
summary(sevenWeight)
summary(sevenTrain)

head(sevenTest)
head(sevenTrain)
head(sevenWeight)

#########################
# Examining how features changed over time
############

valueR <- round(sevenWeight$value,digits=2)
boxplot(valueR ~ sevenWeight$tGroup)

# Examine how the probability changed over epochs, for all training sets & rounds.

par(mfrow=c(1, 3))

dataX <- subset(sevenTrain,outcome == 0)
dataX <- aggregate( .~ epoch,data=dataX,FUN=mean)

dataY <- subset(sevenTrain,outcome == 1)
dataY <- aggregate( .~ epoch,data=dataY,FUN=mean)

dataZ <- subset(sevenTrain,outcome == 2)
dataZ <- aggregate( .~ epoch,data=dataZ,FUN=mean)


plot(probX ~ epoch, data = dataX)
plot(probY ~ epoch, data = dataY)
plot(probZ ~ epoch, data = dataZ)

#########################
# Subset by training & test group
############

subset(sevenTrain,tGroup=="0-2")
subset(sevenTrain,tGroup=="1-0")
subset(sevenTrain,tGroup=="1-2")

# Percent correct, for each test group & all rounds.

perTestCorrect(subset(sevenTest,tGroup=="0"))
perTestCorrect(subset(sevenTest,tGroup=="1"))
perTestCorrect(subset(sevenTest,tGroup=="2"))


#########################
# Misc. functions
############

perTrainCorrect <- function(data) {
  
  correct <- 0
  
  for(a in 1 : length(data[,1])) {
    if(data[a,4] == data[a,5]) {
      correct <- correct + 1
    }
  }
  
  print(correct / length(data[,1]))
  
}

perTestCorrect <- function(data) {
  
  correct <- 0
  
  for(a in 1 : length(data[,1])) {
    if(data[a,3] == data[a,4]) {
      correct <- correct + 1
    }
  }
  
  print(correct / length(data[,1]))
  
}
