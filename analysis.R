
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

dataX <- subset(sevenTrain,outcome == 0)
dataX <- aggregate( .~ epoch,data=dataX,FUN=mean)

dataY <- subset(sevenTrain,outcome == 1)
dataY <- aggregate( .~ epoch,data=dataY,FUN=mean)

dataZ <- subset(sevenTrain,outcome == 2)
dataZ <- aggregate( .~ epoch,data=dataZ,FUN=mean)

par(mfrow=c(1, 3),cex.lab=1.5, cex.axis=1.5, cex.main=1.5, cex.sub=1.5)

plot(probX ~ epoch, data = dataX,lwd=.5,pch = 19,xlab=" ")
#abline(lm(probX ~ epoch, data = dataX), col = "blue",lwd=2.5)

plot(probY ~ epoch, data = dataY,lwd=.5,pch = 19)
#abline(lm(probY ~ epoch, data = dataY), col = "blue",lwd=2.5)

plot(probZ ~ epoch, data = dataZ,lwd=.5,pch = 19,xlab=" ")
#abline(lm(probZ ~ epoch, data = dataZ), col = "blue",lwd=2.5)

#########################
# Subset by training & test group
############

# Percent correct, for each test group & all rounds.

one <- perTestCorrect(subset(sevenTest,tGroup=="0"))
two <- perTestCorrect(subset(sevenTest,tGroup=="1"))
three <- perTestCorrect(subset(sevenTest,tGroup=="2"))

d <- data.frame(
  truePos=c(
    one,
    two,
    three
  ),
  round=c(
    subset(sevenTest,tGroup=="0")$round,
    subset(sevenTest,tGroup=="1")$round,
    subset(sevenTest,tGroup=="2")$round
  ),
  total=c( rep("1",length(sevenTest[,1])) )
)

d <- aggregate( .~ round,data=d,FUN=sum)


par(cex.lab=1.5, cex.axis=1.5, cex.main=1.5, cex.sub=1.5)
plot(density(d$correct/d$total),lwd=2,main=" ")

#########################
# Measure the accuracy
############

accData(sevenTest)

#########################
# Misc. functions
############

perTestCorrect <- function(data) {
  
  correct <- c()
  
  for(a in 1 : length(data[,1])) {
    if(data[a,3] == data[a,4]) {
      correct <- c(correct,1)
    } else {
      correct <- c(correct,0)
    }
  }
  
  #print(correct / length(data[,1]))
  return (correct)
  
}

accData <- function(data) {
  
  d <- matrix(c(rep(0,(3*3))),nrow=3,ncol=3)
  
  for(a in 1 : length(data[,1])) {
    
    one <- data[a,3] + 1
    two <- data[a,4] + 1
    
    d[one,two] <- d[one,two] + 1
    
  }
  
  return (d)
  
}
