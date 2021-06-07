from PIL import Image, ImageOps
from pathlib import Path

path = Path(r"/home/jono/ML/RawData/Jono_2021Masks/202105_OriginalY")

fileList = [imagePath for imagePath in sorted(path.iterdir()) if imagePath.is_file()]

for imageFilename in fileList:
    print("Opening " + str(imageFilename))
    imageFile = Image.open(imageFilename).convert(mode="L")
    inverted = ImageOps.invert(imageFile).convert(mode="1")

    formattedFilename = imageFilename.absolute().parent / imageFilename.name[16:]
    print("Saving to " + str(formattedFilename))
    inverted.save(formattedFilename)

