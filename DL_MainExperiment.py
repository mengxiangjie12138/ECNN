import config
from utils.random5Foldstrain import *
from utils.make_train_test import *

if __name__ == '__main__':


    seed_torch(2021)
    model = "mobileLarge"

    #before begin training plz save your data as data
    #data format should be "data/class1" "data/class2" and so on
    makeCrossValidation = MakeCrossValidationDataFolder(config.crossValidationFolds, 50)
    cvFirstPath = makeCrossValidation.begin_split()
    #Attention !!!!!!!
    #The relative path is relative to the currently running py file,so we must change r"../crossValidationData/...." to r"./crossValidationData/...."
    cvFirstPath = cvFirstPath[1:]
    manyTimesTrain(model, cvFirstPath, times=config.crossValidationFolds, modelName="mobileBigBig")