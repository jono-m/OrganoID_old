from commandline.Program import Program
import argparse
import pathlib
import numpy as np


class Segment(Program):
    def Name(self):
        return "segment"

    def Description(self):
        return "Segment images with the neural network."

    def SetupParser(self, parser: argparse.ArgumentParser):
        parser.add_argument("modelPath", help="Path to trained OrganoID model", type=pathlib.Path)
        parser.add_argument("imagesPath", help="Path to images to analyze.", type=pathlib.Path)
        parser.add_argument("-O", dest="outputPath", default=None,
                            help="Path where analyzed images and data will be saved.",
                            type=pathlib.Path)
        parser.add_argument("-T", dest="threshold", default=-1,
                            type=float,
                            help="If set, the output images will be thresholded as binary images. (0-1)")
        parser.add_argument("--show", action="store_true", help="If set, the output images will be displayed.")

    def RunProgram(self, parserArgs: argparse.Namespace):
        from backend.ImageManager import LoadImages, SaveImage, ShowImage
        from backend.Segmenter import Segmenter
        import matplotlib.pyplot as plt
        images = LoadImages(parserArgs.imagesPath, size=[512, 512])
        segmenter = Segmenter(parserArgs.modelPath)

        count = 1
        for image in images:
            print("Segenting %d: %s" % (count, image.path))
            count += 1
            segmented = segmenter.Segment(image.image)

            if 0 < parserArgs.threshold < 1:
                segmented = segmented > parserArgs.threshold
            else:
                segmented = (plt.get_cmap("hot")(segmented)[:, :, :3] * 255).astype(np.uint8)

            if parserArgs.show:
                ShowImage(segmented)

            if parserArgs.outputPath is not None:
                savePath = parserArgs.outputPath / self.JobName() / image.path.name
                SaveImage(segmented, savePath)
