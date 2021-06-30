from segmentation import DoSegmentation
from pathlib import Path

DoSegmentation(Path(r"C:\Users\jonoj\Documents\ML\TestData\images"),
               Path(r"C:\Users\jonoj\Documents\ML\Segmentations\DevTest"),
               Path(r"C:\Users\jonoj\Documents\ML\Models\OrganoID_train_2021_06_29_23_58_39\trainedModel"), True)
