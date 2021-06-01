from PIL import Image, ImageOps
from pathlib import Path

path = Path(r"/home/jono/ML/RawData/Jono_2021Masks/202105_OriginalY")

fileList = [imagePath for imagePath in sorted(path.iterdir()) if imagePath.is_file()]

for imageFilename in fileList:
    imageFile = Image.open(imageFilename).convert(mode="1")
    inverted = ImageOps.invert(imageFile)

    formattedFilename = imageFilename.absolute().parent / imageFilename.name[16:]

    inverted.save(formattedFilename)

