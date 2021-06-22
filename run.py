from SettingsParser import JobSettings
import segmentation


def DoRun(settings: JobSettings):
    imagesPath = settings.ImagesPath()
    outputPath = settings.OutputPath()
    modelPath = settings.ModelPath()
    useGPU = settings.UseGPU()

    segmentation.DoSegmentation(imagesPath, outputPath, modelPath, useGPU)
