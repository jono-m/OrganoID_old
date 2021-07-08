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
        self.trackingSubparser.add_argument("inputPath", help="Path to images to analyze.", type=pathlib.Path)
        self.trackingSubparser.add_argument("outputPath", help="Path where analyzed images and data will be saved.",
                                            type=pathlib.Path)
        self.trackingSubparser.add_argument("-G", "--useGPU", dest="useGPU", nargs='?', default=False,
                                            type=bool,
                                            help="Set this to True if the GPU should be used")

        self.augmentSubparser = subparsers.add_parser("augment",
                                                      help="Augment images for training.")
        self.augmentSubparser.add_argument("augmentCount", help="Number of augmented images to produce.", type=int)
        self.augmentSubparser.add_argument("-O", "--outputPath", dest='outputPath', nargs='?', default=".",
                                           type=pathlib.Path,
                                           help="Path where the augmented images will be saved.")
        self.augmentSubparser.add_argument("inputPath", help="Path to image and segmentation data. "
                                                             "Directory with subfolders images/ and segmentations/",
                                           type=pathlib.Path)
        self.augmentSubparser.add_argument("-TS", "--testSplit", dest='testSplit', nargs='?', default=0.2,
                                           help="Fraction of images to split for testing (0.0-1.0).", type=float)
        self.augmentSubparser.add_argument("-S" "--size", dest='size', nargs='*', default=[640, 640],
                                           help="Image size to output (e.g. -S 640 640).", type=int)

        self.trainSubparser = subparsers.add_parser("train",
                                                    help="Train the neural network from raw images and manual segmentations.")
        self.trainSubparser.add_argument("-B", "--batch", dest='batchSize', nargs='?', default=1,
                                         help="Number of images to use for a single training pass.", type=int)
        self.trainSubparser.add_argument("-E" "--epochs", dest='epochs', nargs='?', default=1,
                                         help="Number of epochs to train with.", type=int)
        self.trainSubparser.add_argument("-DR" "--dropoutRate", dest='dropoutRate', nargs='?', default=0.2,
                                         help="Dropout rate of CNN during training.", type=float)
        self.trainSubparser.add_argument("-LR" "--learningRate", dest='learningRate', nargs='?', default=0.0001,
                                         help="Neural network learning rate.", type=float)
        self.trainSubparser.add_argument("-P" "--patience", dest='patience', nargs='?', default=5,
                                         help="Steps with no improvement over testing dataset to stop training early.",
                                         type=int)
        self.trainSubparser.add_argument("-S" "--size", dest='size', nargs='*', default=[640, 640],
                                         help="Size of input images (e.g. -S 640 640).", type=int)
        self.trainSubparser.add_argument("-O", "--outputPath", dest='outputPath', nargs='?', default=".",
                                         type=pathlib.Path,
                                         help="Path where the trained model will be saved.")
        self.trainSubparser.add_argument("inputPath", help="Path to image and segmentation data. "
                                                           "Directory with subfolders images/ and segmentations/",
                                         type=pathlib.Path)

        self.args = parser.parse_args()

        self.jobID = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

    def Epochs(self) -> int:
        return self.args.epochs

    def GetBatchSize(self) -> int:
        return self.args.batchSize

    def GetTestSplit(self) -> float:
        return self.args.testSplit

    def AugmentCount(self) -> int:
        return self.args.augmentCount

    def GetSize(self) -> typing.Tuple[int, int]:
        return self.args.size[0], self.args.size[1]

    def GetPatience(self) -> int:
        return self.args.patience

    def GetLearningRate(self) -> float:
        return self.args.learningRate

    def GetDropoutRate(self) -> float:
        return self.args.dropoutRate

    def GetLogPath(self):
        return self.args.logPath

    def ImagesPath(self, critical=True) -> pathlib.Path:
        path = self.args.inputPath / "images"
        if critical and not path.is_dir():
            self.trainSubparser.error(
                "Image path does not exist or is not a directory:  '%s'" % str(path))

        return path.resolve()

    def SegmentationsPath(self, critical=True) -> pathlib.Path:
        path = self.args.inputPath / "segmentations"
        if critical and not path.is_dir():
            self.trainSubparser.error(
                "Segmentation path does not exist or is not a directory:  '%s'" % str(path))
        return path.resolve()

    def TrainingImagesPath(self, critical=True) -> pathlib.Path:
        path = self.args.inputPath / "training" / "images"
        if critical and not path.is_dir():
            self.trainSubparser.error(
                "Image path does not exist or is not a directory:  '%s'" % str(path))

        return path.resolve()

    def TrainingSegmentationsPath(self, critical=True) -> pathlib.Path:
        path = self.args.inputPath / "training" / "segmentations"
        if critical and not path.is_dir():
            self.trainSubparser.error(
                "Segmentation path does not exist or is not a directory:  '%s'" % str(path))
        return path.resolve()

    def TestingImagesPath(self, critical=True):
        path = self.args.inputPath / "testing" / "images"
        if critical and not path.is_dir():
            self.trainSubparser.error(
                "Testing image path does not exist or is not a directory:  '%s'" % str(path))

        return path.resolve()

    def TestingSegmentationsPath(self, critical=True):
        path = self.args.inputPath / "testing" / "segmentations"
        if critical and not path.is_dir():
            self.trainSubparser.error(
                "Testing segmentations path does not exist or is not a directory:  '%s'" % str(path))

        return path.resolve()

    def ModelPath(self) -> pathlib.Path:
        return self.args.modelPath.resolve()

    def UseGPU(self) -> bool:
        return self.args.useGPU

    def OutputPath(self) -> pathlib.Path:
        return self.args.outputPath.resolve() / ("OrganoID_" + self.GetMode() + "_" + self.jobID)

    def GetMode(self):
        return self.args.subparser_name
