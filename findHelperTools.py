import itertools

# Used to remove all features with non numerical values. this function works explicity with numbers
def removeNonNumbers(dataset):
    dataset.dropna(axis='index', how='any', inplace=True)
    allFeatures = list(dataset.columns.values) #grab a list of all features
    types = list(dataset.dtypes)
    featuresAndTypes = list(zip(allFeatures,types))
    dropThem = list([])
    for y in featuresAndTypes:
        #print(types[y])
        if (y[1] == "object"):
            #print(y[1])
            dropThem.append(y[0])
            #print(dropThem)
    #print(dropThem)
    dataset.drop(labels=dropThem, axis='columns', inplace=True)
    return dataset

#gets all possibilities of combonations of features with a user defined limit to the max amount of features
def getPossibilities(ListOfFeatures,maxAmtOfFeatures=9999999 ):
    if (maxAmtOfFeatures == 9999999):
        #print("works")
        maxAmtOfFeatures = len(ListOfFeatures)-1
    #get all possible combinations of features
    allPossibilities = []
    for z in range(1,maxAmtOfFeatures):
        combos = list(itertools.combinations(ListOfFeatures,z))
        for i in combos:
            allPossibilities.append(i)
    return allPossibilities

def percentBucketMaker(x):
    if x >=.75:
        return 4
    if .75>=x>.5:
        return 3
    if .5>=x>.25:
        return 2
    if .25>=x>0:
        return 1
    
#returns a list of all features in a given dataset 
def grabFeatures(dataset):
    return list(dataset.columns.values)

