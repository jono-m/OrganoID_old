from SettingsParser import JobSettings

settings = JobSettings()

print("Beginning OrganoID job " + settings.jobID + "...")

if settings.GetMode() == "train":
    import training

    training.DoTraining(settings)
elif settings.GetMode() == "run":
    import run

    run.DoRun(settings)
elif settings.GetMode() == "augment":
    import augmentation

    augmentation.DoAugment(settings)
elif settings.GetMode() == "monitor":
    import monitoring

    monitoring.DoMonitor(settings)

print("Job complete.")
