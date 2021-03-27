from datetime import datetime
from SettingsParser import JobSettings

settings = JobSettings()

print("Beginning OrganoID job " + settings.jobID + "...")

if settings.GetMode() == "train":
    import training

    training.DoTraining(settings)
elif settings.GetMode() == "run":
    import run

    run.DoRun(settings)

print("Job complete.")
