import argparse
import pathlib
import datetime
import typing


class JobSettings:
    def __init__(self):
        parser = argparse.ArgumentParser(
            description="Neural network segmentation and tracking of organoid microscopy images.")

        subparsers = parser.add_subparsers(dest="subparser_name")

        self.trackingSubparser = subparsers.add_parser("run", help="Pipeline execution commands.")

        self.trackingSubparser.add_argument("modelPath", help="Path to trained OrganoID model", type=pathlib.Path)
        self.trackingSubparser.add_argument("imagePath", help="Path to images to analyze.", type=pathlib.Path)
        self.trackingSubparser.add_argument("outputPath", help="Path where analyzed images and data will be saved.",
                                            type=pathlib.Path)

        self.trainSubparser = subparsers.add_parser("train",
                                                    help="Train the neural network from raw images and manual segmentations.")

        self.trainSubparser.add_argument("-A", "--augment", dest='augmentSize', nargs='?', default=0,
                                         help="Number of augmented images to produce (0 for no augmentation).",
                                         type=int)
        self.trainSubparser.add_argument("-B", "--batch", dest='batchSize', nargs='?', default=1,
                                         help="Number of images to use for a single training pass.", type=int)
        self.trainSubparser.add_argument("-TS", "--testSplit", dest='testSplit', nargs='?', default=0.8,
                                         help="Fraction of images to use for testing (0.0-1.0).", type=float)
        self.trainSubparser.add_argument("-E" "--epochs", dest='epochs', nargs='?', default=1,
                                         help="Number of epochs to train with.", type=int)
        self.trainSubparser.add_argument("-S" "--size", dest='size', nargs='*', default=[640, 640],
                                         help="Size of input images (e.g. -S 640 640).", type=int)
        self.trainSubparser.add_argument("-F", "--fit", dest='doFit', action='store_true',
                                         help="Fit model to training data.")
        self.trainSubparser.add_argument("-O", "--outputPath", dest='outputPath', nargs='?', default=".",
                                         type=pathlib.Path,
                                         help="Path where the processed images and/or trained model will be saved.")
        self.trainSubparser.add_argument("imagePath", help="Path to image training data.", type=pathlib.Path)
        self.trainSubparser.add_argument("segmentationPath", help="Path to segmentation training data.",
                                         type=pathlib.Path)

        self.args = parser.parse_args()

        self.jobID = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

    def Epochs(self) -> int:
        return self.args.epochs

    def GetBatchSize(self) -> int:
        return self.args.batchSize

    def GetTestSplit(self) -> float:
        return self.args.testSplit

    def ShouldAugment(self) -> bool:
        return self.args.augmentSize > 0

    def AugmentSize(self) -> int:
        return self.args.augmentSize

    def GetSize(self) -> typing.Tuple[int, int]:
        return self.args.size[0], self.args.size[1]

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

    def OutputPath(self) -> pathlib.Path:
        return self.args.outputPath.resolve() / ("OrganoID_" + self.jobID)

    def GetMode(self):
        return self.args.subparser_name
