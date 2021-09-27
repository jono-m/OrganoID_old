from commandline.Program import Program
import argparse
import pathlib


class LearningVis(Program):
    def Name(self):
        return "learningvis"

    def Description(self):
        return "Visualize network learning with a folder of epochs."

    def SetupParser(self, parser: argparse.ArgumentParser):
        parser.add_argument("modelPath", help="Path to directory of epoch checkpoint models", type=pathlib.Path)
        parser.add_argument("imagePath", help="Path to image to analyze.", type=pathlib.Path)
        parser.add_argument("-O", dest="outputPath", default=None,
                            help="Path where learning visualization will be saved.",
                            type=pathlib.Path)

    def RunProgram(self, parserArgs: argparse.Namespace):
        from backend.ImageManager import LoadImages, ContrastOp, SaveGIF, sort_paths_nicely
        from backend.Segmenter import Segmenter
        import numpy as np
        import matplotlib.pyplot as plt
        images = list(LoadImages(parserArgs.imagePath, size=[512, 512], mode="L"))
        image = images[0]
        count = 1

        modelPaths = list(parserArgs.modelPath.iterdir())
        sort_paths_nicely(modelPaths)

        heatMaps = []

        for modelPath in modelPaths:
            print("Segenting %d: %s" % (count, modelPath))
            segmenter = Segmenter(modelPath)
            count += 1

            segmented_raw = image.DoOperation(ContrastOp).DoOperation(segmenter.Segment)

            heatMap = segmented_raw.DoOperation(lambda x: (plt.get_cmap("hot")(x)[:, :, :3] * 255).astype(np.uint8))
            heatMaps.append(heatMap.frames[0])

        if parserArgs.outputPath is not None:
            SaveGIF(heatMaps, parserArgs.outputPath / (image.path.stem + ".gif"))