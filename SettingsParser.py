import argparse
import pathlib
import datetime


class JobSettings:
    def __init__(self):
        parser = argparse.ArgumentParser(
            description="Neural network segmentation and tracking of organoid microscopy images.")

        subparsers = parser.add_subparsers(dest="subparser_name")

        self.trainSubparser = subparsers.add_parser("train", help="Training commands.")

        self.trainSubparser.add_argument("-A", "--augment", dest='doAugment', action='store_true',
                                         help="Augment training data.")
        self.trainSubparser.add_argument("-P", "--preprocess", dest='doPreprocess', action='store_true',
                                         help="Preprocess training data.")
        self.trainSubparser.add_argument("-F", "--fit", dest='doFit', action='store_true',
                                         help="Fit model to training data.")
        self.trainSubparser.add_argument("-M", "--modelPath", dest='modelPath', nargs='?', default=".",
                                         type=pathlib.Path)
        self.trainSubparser.add_argument("imagePath", help="Path to image training data.", type=pathlib.Path)
        self.trainSubparser.add_argument("segmentationPath", help="Path to segmentation training data.",
                                         type=pathlib.Path)

        self.args = parser.parse_args()

        self.jobID = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

    def ShouldPreprocess(self) -> bool:
        return self.args.doPreprocess

    def ShouldAugment(self) -> bool:
        return self.args.doAugment

    def ShouldFit(self):
        return self.args.doFit

    def ImagesPath(self) -> pathlib.Path:
        if not self.args.imagePath.is_dir():
            self.trainSubparser.error(
                "Image path does not exist or is not a directory:  '%s'" % str(self.args.imagePath))

        return self.args.imagePath.resolve()

    def SegmentationsPath(self) -> pathlib.Path:
        if not self.args.segmentationPath.is_dir():
            self.trainSubparser.error(
                "Segmentation path does not exist or is not a directory:  '%s'" % str(self.args.segmentationPath))
        return self.args.segmentationPath.resolve()

    def ModelPath(self) -> pathlib.Path:
        return self.args.modelPath.resolve()

    def GetMode(self):
        return self.args.subparser_name

    def __repr__(self):
        return """
        OrganoID settings:
        -----
        Preprocess: %s,
        Augment: %s,
        Images Path: %s,
        Segmentations Path: %s,
        Model Path: %s
        """ % (self.ShouldPreprocess(),
               self.ShouldAugment(),
               self.ImagesPath(),
               self.SegmentationsPath(),
               self.ModelPath())
