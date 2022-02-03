import sys

from commandline.Program import Program
import argparse
import pathlib
from util import Printer


class Analyze(Program):
    def Name(self):
        return "analyze"

    def Description(self):
        return "Analyze organoids in a labeled image."

    def SetupParser(self, parser: argparse.ArgumentParser):
        parser.add_argument("imagesPath", help="Path to labeled images.", type=pathlib.Path)
        parser.add_argument("outputPath", help="Directory where results will be saved.", type=pathlib.Path)
        parser.add_argument("measurements", nargs="+", help="List of features to measure. "
                                                            "See https://scikit-image.org/docs/dev/api/skimage.measure.html#skimage.measure.regionprops for available features")

    def RunProgram(self, parserArgs: argparse.Namespace):
        from backend.ImageManager import LoadImages
        from skimage.measure import regionprops

        # Load images
        images = LoadImages(parserArgs.imagesPath, size=[512, 512])

        features = parserArgs.measurements
        count = 1
        singleAnalysisOutput = "Image name, Organoid label, " + ", ".join(features) + "\n"
        for image in images:
            for i, frame in enumerate(image.frames):
                if len(image.frames) > 1:
                    name = image.path.stem + "_" + str(i)
                else:
                    name = image.path.stem
                Printer.printRep("Analyzing image %d" % count)
                rps = regionprops(frame)
                for rp in rps:
                    data = ", ".join([str(eval("rp.%s" % feature, {}, {"rp": rp})) for feature in features])

                    singleAnalysisOutput += "%s, %s, %s\n" % (name, rp.label, data)
                count += 1
        Printer.printRep()
        path: pathlib.Path = parserArgs.outputPath
        path.mkdir(parents=True, exist_ok=True)
        csvFile = open(parserArgs.outputPath / "singleOrganoidMeasurements.csv", 'w+')
        csvFile.write(singleAnalysisOutput)
        csvFile.close()

