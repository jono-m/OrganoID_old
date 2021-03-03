from SettingsParser import JobSettings


def DoTraining(settings: JobSettings):
    trainingImagesPath = settings.ImagesPath()
    trainingSegmentationsPath = settings.SegmentationsPath()
    modelSavePath = settings.ModelPath()

    if settings.ShouldPreprocess():
        import preprocessing
        trainingImagesPath, trainingSegmentationsPath = preprocessing.PreprocessImages(settings.jobID,
                                                                                       trainingImagesPath,
                                                                                       trainingSegmentationsPath)

    if settings.ShouldAugment():
        import augmentation
        trainingImagesPath, trainingSegmentationsPath = augmentation.AugmentImages(settings.jobID,
                                                                                   trainingImagesPath,
                                                                                   trainingSegmentationsPath,
                                                                                   10)

    if settings.ShouldFit():
        import modelFitting
        modelFitting.FitModel(settings.jobID, trainingImagesPath, trainingSegmentationsPath, modelSavePath, epochs=1,
                              test_size=0.2, patience=5)
