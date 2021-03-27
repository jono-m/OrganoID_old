from SettingsParser import JobSettings


def DoTraining(settings: JobSettings):
    trainingImagesPath = settings.ImagesPath()
    trainingSegmentationsPath = settings.SegmentationsPath()
    modelSavePath = settings.OutputPath()

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
                                                                                   settings.AugmentSize())

    if settings.ShouldFit():
        import modelFitting
        modelFitting.FitModel(settings.jobID, trainingImagesPath, trainingSegmentationsPath, modelSavePath, epochs=settings.Epochs(),
                              test_size=settings.GetTestSplit(), patience=1, batch_size=settings.GetBatchSize())
