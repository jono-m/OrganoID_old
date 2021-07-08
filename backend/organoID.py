from SettingsParser import JobSettings

settings = JobSettings()

print("Beginning OrganoID job " + settings.jobID + "...")

if settings.GetMode() == "train":
    import training

    training.FitModel(settings.TrainingImagesPath(), settings.TrainingSegmentationsPath(), settings.TestingImagesPath(),
                      settings.TestingSegmentationsPath(), settings.OutputPath(), settings.Epochs(),
                      settings.GetBatchSize(), settings.GetPatience(), settings.GetSize(), settings.GetDropoutRate(),
                      settings.GetLearningRate())

elif settings.GetMode() == "run":
    import segmentation

    segmentation.DoSegmentation(settings.ImagesPath(), settings.OutputPath(), settings.ModelPath(), settings.UseGPU())
elif settings.GetMode() == "augment":
    import augmentation

    augmentation.Augment(settings.OutputPath(),
                         settings.ImagesPath(),
                         settings.SegmentationsPath(),
                         settings.GetTestSplit(),
                         settings.GetSize(),
                         settings.AugmentCount())

print("Job complete.")
