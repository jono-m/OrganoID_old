from SettingsParser import JobSettings
import segmentation


def DoRun(settings: JobSettings):
    imagesPath = settings.ImagesPath()
    outputPath = settings.OutputPath()
    modelPath = settings.ModelPath()

    segmentation.DoSegmentation(imagesPath, outputPath, modelPath)
