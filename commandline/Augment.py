# Augment.py -- Sub-program to augment training data.

from commandline.Program import Program
import argparse
import pathlib


class Augment(Program):
    def Name(self):
        return "augment"

    def Description(self):
        return "Augment images for training."

    def SetupParser(self, parser: argparse.ArgumentParser):
        parser.add_argument("inputPath", help="Path to image and segmentation data. "
                                              "Directory should have the subfolders images/ and segmentations/",
                            type=pathlib.Path)
        parser.add_argument("outputPath", type=pathlib.Path, help="Path where the augmented images will be saved.")
        parser.add_argument("augmentCount", help="Number of augmented images to produce.", type=int)

    def RunProgram(self, parserArgs: argparse.Namespace):
        from backend.Augment import Augment
        Augment(parserArgs.inputPath / "images",
                parserArgs.inputPath / "segmentations",
                parserArgs.outputPath,
                parserArgs.augmentCount)
