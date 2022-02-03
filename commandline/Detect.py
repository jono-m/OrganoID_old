# Detect.py -- detects organoids in a microscopy image.
from commandline.Program import Program
import argparse
import pathlib


class Detect(Program):
    def Name(self):
        return "detect"

    def Description(self):
        return "Detect organoids in microscopy images."

    def SetupParser(self, parser: argparse.ArgumentParser):
        parser.add_argument("modelPath", help="Path to trained OrganoID model", type=pathlib.Path)
        parser.add_argument("imagesPath", help="Path to images to analyze.", type=pathlib.Path)
        parser.add_argument("-O", dest="outputPath", default=None,
                            help="Directory where detection images will be saved.",
                            type=pathlib.Path)
        parser.add_argument("--heat", action="store_true", help="If set, the output images will also be produced in a "
                                                                "heatmap format, which is good for visualizing "
                                                                "detections.")
        parser.add_argument("--show", action="store_true", help="If set, the output images will be displayed.")

    def RunProgram(self, parserArgs: argparse.Namespace):
        from backend.ImageManager import LoadImages, SaveImage, ShowImage, SaveTIFFStack
        from backend.Detector import Detector

        # Load the images
        images = LoadImages(parserArgs.imagesPath, size=[512, 512], mode="L")

        # Load neural network detector
        detector = Detector(parserArgs.modelPath)

        count = 1
        outputImages = []

        for image in images:
            print("Detecting %d: %s" % (count, image.path))
            count += 1

            detected_raw = image.DoOperation(detector.Detect, True)
            if parserArgs.outputPath is not None:
                if len(detected_raw.frames) > 1:
                    extension = ".tiff"
                else:
                    extension = image.path.suffix
                fileName = image.path.stem + "_detected" + extension
                savePath = parserArgs.outputPath / fileName
                if len(detected_raw.frames) > 1:
                    SaveTIFFStack(detected_raw.frames, savePath)
                else:
                    SaveImage(detected_raw.frames[0], savePath)
            if parserArgs.heat:
                heat = detected_raw.DoOperation(lambda x: detector.ConvertToHeatmap(x))
                outputImages.append(heat)
                if parserArgs.outputPath is not None:
                    if len(heat.frames) > 1:
                        extension = ".tiff"
                    else:
                        extension = ".png"
                    fileName = image.path.stem + "_heat" + extension
                    savePath = parserArgs.outputPath / fileName
                    if len(heat.frames) > 1:
                        SaveTIFFStack(heat.frames, savePath)
                    else:
                        SaveImage(heat.frames[0], savePath)

        if parserArgs.show:
            for outputImage in outputImages:
                for frame in outputImage.frames:
                    ShowImage(frame)
