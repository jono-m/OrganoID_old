from commandline.Program import Program
import argparse
import pathlib


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
        parser.add_argument("-T", dest="threshold", default=False,
                            type=float,
                            help="Set to threshold images to binary images. (0-1)")
        parser.add_argument("--raw", action="store_true", help="If set, the output images will also be produced in raw "
                                                               "format, which is needed for postprocessing.")
        parser.add_argument("--heat", action="store_true", help="If set, the output images will also be produced in a "
                                                                "heatmap format, which is good for visualizing raw "
                                                                "network behavior.")
        parser.add_argument("--show", action="store_true", help="If set, the output images will be displayed.")

    def RunProgram(self, parserArgs: argparse.Namespace):
        from backend.ImageManager import LoadImages, SaveImage, ShowImage
        from backend.Segmenter import Segmenter
        import numpy as np
        import matplotlib.pyplot as plt
        images = LoadImages(parserArgs.imagesPath, size=[512, 512], mode="L")
        segmenter = Segmenter(parserArgs.modelPath)

        count = 1
        for image in images:
            print("Segenting %d: %s" % (count, image.path))
            count += 1

            segmented_raw = segmenter.Segment(image.image)

            outputImages = []

            if parserArgs.threshold:
                outputImages.append(("threshold", segmented_raw > parserArgs.threshold, None))
            if parserArgs.heat:
                outputImages.append(
                    ("heat", (plt.get_cmap("hot")(segmented_raw)[:, :, :3] * 255).astype(np.uint8), None))
            if parserArgs.raw:
                outputImages.append(("raw", segmented_raw, ".tiff"))

            if parserArgs.show:
                [ShowImage(outputImage) for (_, outputImage, _) in outputImages]

            if parserArgs.outputPath is not None:
                for name, outputImage, extension in outputImages:
                    if extension is None:
                        extension = image.path.suffix
                    fileName = image.path.stem + "_" + name + extension
                    savePath = parserArgs.outputPath / self.JobName() / fileName
                    print(savePath)
                    SaveImage(outputImage, savePath)
