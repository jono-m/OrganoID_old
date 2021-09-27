from pathlib import Path
from backend.ImageManager import LoadImages, LabelToRGB
from PIL import Image, ImageDraw, ImageFont
from skimage.measure import regionprops, label

directory = Path(r"C:\Users\jonoj\Downloads\pipeline_2021_09_09_12_22_27\pipeline_2021_09_09_12_22_27\*raw*")
outputDirectory = Path(
    r"C:\Users\jonoj\Downloads\pipeline_2021_09_09_12_22_27\pipeline_2021_09_09_12_22_27\AnalysisOut")
images = LoadImages(directory)

outputDirectory.mkdir(parents=True, exist_ok=True)
font = ImageFont.truetype("arial.ttf", 16)
for image in images:
    labeledImage = label(image.frames[0])
    rgbImage = LabelToRGB(labeledImage)
    regionProperties = regionprops(labeledImage)

    withTextImage = Image.fromarray(rgbImage)
    drawer = ImageDraw.Draw(withTextImage)
    for regionProperty in regionProperties:
        (y, x) = regionProperty.centroid
        drawer.text((x, y), str(regionProperty.label), anchor="ms", fill=(255, 255, 255, 255), font=font)
    savePath = outputDirectory / image.path.name
    withTextImage.save(savePath)

    with open(outputDirectory / ("AREAS_" + image.path.stem + ".csv"), 'w', newline='') as csvfile:
        csvfile.write(
            "Organoid ID, Area\n")
        for regionProperty in regionProperties:
            csvfile.write("%d, %d\n" %
                          (regionProperty.label, regionProperty.area))
