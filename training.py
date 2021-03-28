from SettingsParser import JobSettings
import augmentation


def DoTraining(settings: JobSettings):
    settings.OutputPath().mkdir()

    trainingImagesPath, trainingSegmentationsPath = augmentation.AugmentImages(settings.ImagesPath(),
                                                                               settings.SegmentationsPath(),
                                                                               settings.OutputPath(),
                                                                               settings.AugmentSize(),
                                                                               settings.GetSize())

    if settings.ShouldFit():
        import modelFitting
        modelFitting.FitModel(trainingImagesPath, trainingSegmentationsPath, settings.OutputPath(),
                              epochs=settings.Epochs(),
                              test_size=settings.GetTestSplit(), patience=1, batch_size=settings.GetBatchSize(),
                              imageSize=settings.GetSize())
