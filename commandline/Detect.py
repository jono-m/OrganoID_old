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
        import numpy as np
        import matplotlib.pyplot as plt

        # Load the images
        images = LoadImages(parserArgs.imagesPath, size=[512, 512], mode="L")

        # Load neural network detector
        detector = Detector(parserArgs.modelPath)

        count = 1
        outputImages = []

        for image in images:
            print("Detecting %d: %s" % (count, image.path))
            count += 1

            detected_raw = image.DoOperation(detector.Detect)

            if parserArgs.heat:
                # Color pixels with a heatmap.
                outputImages.append(
                    ("heat",
                     detected_raw.DoOperation(lambda x: (plt.get_cmap("hot")(x)[:, :, :3] * 255).astype(np.uint8)),
                     None))
            outputImages.append(("detected", detected_raw, ".tiff"))

        if parserArgs.show:
            for (_, outputImage, _) in outputImages:
                for frame in outputImage.frames:
                    ShowImage(frame)

        if parserArgs.outputPath is not None:
            for name, outputImage, extension in outputImages:
                if extension is None:
                    extension = outputImage.path.suffix
                if len(outputImage.frames) > 1:
                    extension = ".tiff"
                fileName = outputImage.path.stem + "_" + name + extension
                savePath = parserArgs.outputPath / fileName
                print(savePath)
                if len(outputImage.frames) > 1:
                    SaveTIFFStack(outputImage.frames, savePath)
                else:
                    SaveImage(outputImage.frames[0], savePath)
