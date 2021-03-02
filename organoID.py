import augmentation
import preprocessing
import training
from datetime import datetime

trainingImagesPath = "C:/Users/jonoj/Documents/ML/2019_originalX"
trainingSegmentationsPath = "C:/Users/jonoj/Documents/ML/2019_originalY"
modelSavePath = "C:/Users/jonoj/Documents/ML/Models"

jobID = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

print("Beginning OrganoID job " + jobID + "...")

trainingImagesPath, trainingSegmentationsPath = preprocessing.PreprocessImages(jobID,
                                                                               trainingImagesPath,
                                                                               trainingSegmentationsPath)

trainingImagesPath, trainingSegmentationsPath = augmentation.AugmentImages(jobID,
                                                                           trainingImagesPath,
                                                                           trainingSegmentationsPath,
                                                                           10)

modelPath = training.TrainModel(jobID, trainingImagesPath, trainingSegmentationsPath, modelSavePath, epochs=1,
                                test_size=0.2, patience=5)

print("Job complete.")
