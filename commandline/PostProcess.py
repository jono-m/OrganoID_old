from commandline.Program import Program
import argparse
import pathlib
import numpy as np


class PostProcess(Program):
    def Name(self):
        return "postprocess"

    def Description(self):
        return "Post process images that were produced from the CNN."

    def SetupParser(self, parser: argparse.ArgumentParser):
        parser.add_argument("imagesPath", help="Path to raw images to process.", type=pathlib.Path)
        parser.add_argument("-O", dest="outputPath", default=None,
                            help="Path where processed images will be saved.",
                            type=pathlib.Path)
        parser.add_argument("-T", dest="threshold", default=0.5,
                            type=float,
                            help="Segmentation threshold (0-1).")
        parser.add_argument("-W", dest="watershedThresh", default=False,
                            type=float,
                            help="Watershed threshold (0-1).")
        parser.add_argument("-A", dest="minArea", default=20,
                            type=int,
                            help="Area cutoff.")
        parser.add_argument("-B", dest="borderCutoff", default=None,
                            type=float,
                            help="Remove organoids that are touching borders with more than [BORDERCUTOFF]*Diameter "
                                 "pixels (0-1).")
        parser.add_argument("--raw", action="store_true",
                            help="If set, a raw-labeled version of each image will be produced.")
        parser.add_argument("--rgb", action="store_true", help="If set, an RGB version of each image will be produced.")
        parser.add_argument("--show", action="store_true", help="If set, the output images will be displayed.")

    def RunProgram(self, parserArgs: argparse.Namespace):
        from backend.ImageManager import LoadImages, SaveImage, ShowImage, LabelToRGB
        from backend.Label import Label
        from backend.PostProcessing import PostProcess
        images = LoadImages(parserArgs.imagesPath, size=[512, 512])

        count = 1
        for image in images:
            print("Processing %d: %s" % (count, image.path))
            count += 1

            labeled = Label(image.image, parserArgs.threshold, parserArgs.watershedThresh)

            postProcessed = PostProcess(labeled, parserArgs.minArea, parserArgs.borderCutoff)

            outputImages = []

            if parserArgs.rgb:
                outputImages.append(("rgb", LabelToRGB(postProcessed)))
            if parserArgs.raw:
                outputImages.append(("raw", postProcessed))

            if parserArgs.show:
                [ShowImage(outputImage) for (_, outputImage) in outputImages]

            if parserArgs.outputPath is not None:
                for name, outputImage in outputImages:
                    savePath = parserArgs.outputPath / self.JobName() / (
                            image.path.stem + "_" + name + image.path.suffix)
                    SaveImage(outputImage, savePath)
