from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score, f1_score
from sklearn.svm import SVC

def LinearRegressionClassifier(x,y,bestR2,bestR2MSE,bestXY,index,allPossibleFeatureCombos,singleXFeature,oldReg):
            reg = LinearRegression()
            reg.fit(x,y)
            yPrediction = reg.predict(x)
            returnList = []
            returnList.append(oldReg)
            returnList.append(bestR2)
            returnList.append(bestR2MSE)
            returnList.append(bestXY)
            #capture best scores
            if (reg.score(x,y)==bestR2):
                if (mean_squared_error(y,yPrediction) < bestR2MSE):
                    bestR2 = reg.score(x,y)
                    bestR2MSE =mean_squared_error(y,yPrediction)
                    bestXY = [allPossibleFeatureCombos[index],singleXFeature]
                    returnList = []
                    returnList.append(reg)
                    returnList.append(bestR2)
                    returnList.append(bestR2MSE)
                    returnList.append(bestXY)
            if(reg.score(x,y)>bestR2):
                bestR2 = reg.score(x,y)
                bestR2MSE =mean_squared_error(y,yPrediction)
                bestXY = [allPossibleFeatureCombos[index],singleXFeature]
                returnList = []
                returnList.append(reg)
                returnList.append(bestR2)
                returnList.append(bestR2MSE)
                returnList.append(bestXY)

                
            return returnList

def SVMClassifier(x,y,bestAccuracyScore,bestF1Score,bestXY,index,allPossibleFeatureCombos,singleXFeature,oldReg,myXes,dataSetTest):#needs buckets
        reg = SVC(random_state = 29, max_iter=10, tol=0.1)
        reg.fit(x,y)
        xTest = dataSetTest[myXes]
        y = dataSetTest["bucket" + str(singleXFeature)]
        yPrediction = reg.predict(xTest)
        returnList = []
        returnList.append(oldReg)
        returnList.append(bestAccuracyScore)
        returnList.append(bestF1Score)
        returnList.append(bestXY)
        #capture best scores
        #svc scores are accracy precision sensitiity and f1
        if (accuracy_score(y, yPrediction)==bestAccuracyScore):
            if (f1_score(y, yPrediction, average="weighted") < bestF1Score):
                bestAccuracyScore =accuracy_score(y, yPrediction)
                bestF1Score =f1_score(y, yPrediction, average="weighted")
                bestXY = [allPossibleFeatureCombos[index],singleXFeature]
                returnList = []
                returnList.append(reg)
                returnList.append(bestAccuracyScore)
                returnList.append(bestF1Score)
                returnList.append(bestXY)
    
        if(accuracy_score(y, yPrediction)>bestAccuracyScore):
            bestAccuracyScore =accuracy_score(y, yPrediction)
            bestF1Score =f1_score(y, yPrediction, average="weighted")
            bestXY = [allPossibleFeatureCombos[index],singleXFeature]
            returnList = []
            returnList.append(reg)
            returnList.append(bestAccuracyScore)
            returnList.append(bestF1Score)
            returnList.append(bestXY)
        return returnList


def BinaryClassifier(x,y,bestAccuracyScore,bestF1Score,bestXY,index,allPossibleFeatureCombos,singleXFeature):#needs buckets
    reg = BinaryClassifier(random_state = 29, max_iter=10, tol=0.1)
    reg.fit(x,y)
    yPrediction = reg.predict(x)#use xs from testset
    #capture best scores
    #svc scores are accracy precision sensitiity and f1
    if (accuracy_score(y_train, yPrediction)==bestAccuracyScore):
        if (f1_score(y_train, yPrediction, average="weighted") < bestF1Score):
            bestAccuracyScore =accuracy_score(y_train, yPrediction)
            bestF1Score =f1_score(y_train, yPrediction, average="weighted")
            bestXY = [allPossibleFeatureCombos[index],singleXFeature]

    if(accuracy_score(y_train, yPrediction)>bestAccuracyScore):
        bestAccuracyScore =accuracy_score(y_train, yPrediction)
        bestF1Score =f1_score(y_train, yPrediction, average="weighted")
        bestXY = [allPossibleFeatureCombos[index],singleXFeature]
    return reg
