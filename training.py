from SettingsParser import JobSettings
import modelFitting


def DoTraining(settings: JobSettings):
    settings.OutputPath().mkdir()
    modelFitting.FitModel(settings.ImagesPath(), settings.SegmentationsPath(), settings.OutputPath(),
                          epochs=settings.Epochs(), test_size=settings.GetTestSplit(), patience=1,
                          batch_size=settings.GetBatchSize(), imageSize=settings.GetSize())
