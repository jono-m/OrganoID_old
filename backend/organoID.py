from SettingsParser import JobSettings

settings = JobSettings()

print("Beginning OrganoID job " + settings.jobID + "...")

if settings.GetMode() == "train":
    import training

    training.FitModel(settings.InputPath("/training/images/"),
                      settings.InputPath("/training/segmentations/"),
                      settings.InputPath("/testing/images/"),
                      settings.InputPath("/testing/segmentations/"),
                      settings.OutputPath(), settings.Epochs(),
                      settings.GetBatchSize(), settings.GetPatience(), settings.GetSize(), settings.GetDropoutRate(),
                      settings.GetLearningRate())

elif settings.GetMode() == "segment":
    import segmentation

    segmentation.DoSegmentation(settings.InputPath(), settings.OutputPath(), settings.ModelPath(), settings.UseGPU(),
                                settings.Threshold())
elif settings.GetMode() == "augment":
    import augmentation

    augmentation.Augment(settings.OutputPath(),
                         settings.InputPath("/images/"),
                         settings.InputPath("/segmentations/"),
                         settings.GetTestSplit(),
                         settings.GetSize(),
                         settings.AugmentCount())

print("Job complete.")
