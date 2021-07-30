from commandline.Program import Program
import argparse
import pathlib


class Split(Program):
    def Name(self):
        return "split"

    def Description(self):
        return "Split up images for training."

    def SetupParser(self, parser: argparse.ArgumentParser):
        parser.add_argument("inputPath", help="Path to image and segmentation data. "
                                              "Directory should have the subfolders images/ and segmentations/",
                            type=pathlib.Path)
        parser.add_argument("-O", "--outputPath", dest='outputPath', default=".",
                            type=pathlib.Path,
                            help="Path where the split images will be saved.")
        parser.add_argument("-TS", "--testSplit", dest='testSplit', default=0.2,
                            help="Fraction of images to split for testing (0.0-1.0).", type=float)

    def RunProgram(self, parserArgs: argparse.Namespace):
        from backend.SplitData import SplitData
        SplitData((parserArgs.inputPath / "images").iterdir(), (parserArgs.inputPath / "segmentations").iterdir(),
                  parserArgs.testSplit, parserArgs.outputPath / self.JobName())
