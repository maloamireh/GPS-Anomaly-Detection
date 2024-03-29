#Read in data
ka.local07groundtruth <- read.csv("C:/Users/camer/Downloads/ka-local07groundtruth.csv")
ka.local07attack <- read.csv("C:/Users/camer/Downloads/ka-local07attack.csv")

#Open necessary libraries
library(corrr)
library(ggcorrplot)
library(FactoMineR)
library(factoextra)

#Normalize data, create the correlation matrix and plot correlations
normalData = scale(ka.local07groundtruth)
corrMatrix = cor(normalData)
ggcorrplot(corrMatrix)

#Apply PCA and view the summary
pca = princomp(corrMatrix)
summary(pca)

#Scree plot
fviz_eig(pca, addlabels = TRUE)

#Biplot
fviz_pca_var(pca, col.var = "black")

#Contribution of each attribute measured by square cosine (components 1 through 4)
fviz_cos2(pca, choice = "var", axes = 1:4)

#From here may need to determine cutoff for number of components, and ultimately which features should be selected