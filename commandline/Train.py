# Train.py -- sub-program to train the neural network.

from commandline.Program import Program
import argparse
import pathlib


class Train(Program):
    def Name(self):
        return "train"

    def Description(self):
        return "Train the neural network from raw images and manual segmentations."

    def SetupParser(self, parser: argparse.ArgumentParser):
        parser.add_argument("inputPath", help="Path to image and segmentation data. "
                                              "Directory with subfolders training/ and validation/ with "
                                              "respective subfolders images/ and segmentations/",
                            type=pathlib.Path)
        parser.add_argument("outputPath", type=pathlib.Path, help="Path where the trained model will be saved.")
        parser.add_argument("-B", dest='batchSize', default=1,
                            help="Number of images to use for a single training pass.", type=int)
        parser.add_argument("-E", dest='epochs', default=1,
                            help="Max number of times that the full image dataset should be passed through.", type=int)
        parser.add_argument("-DR", dest='dropoutRate', default=0.2,
                            help="Dropout rate of CNN during training.", type=float)
        parser.add_argument("-LR", dest='learningRate', default=0.0001,
                            help="Neural network learning rate.", type=float)
        parser.add_argument("-P", dest='patience', default=5,
                            help="Steps with no improvement over validation dataset to stop training early.",
                            type=int)
        parser.add_argument("-S", dest='size', nargs=2, default=[512, 512],
                            help="Size of input images (e.g. -S 512 512).", type=int)

    def RunProgram(self, parserArgs: argparse.Namespace):
        from backend.ModelTrainer import ModelTrainer
        from backend.ModelDataGenerator import ModelDataGenerator

        # OrganoID U-Net starts with 8 filters in the first convolutional layer. Create the trainer.
        trainer = ModelTrainer(parserArgs.size, parserArgs.dropoutRate, 8)

        # Training and validation images are read in on-the-fly to avoid storing everything in memory.
        trainingDataGenerator = ModelDataGenerator(parserArgs.inputPath / "training" / "images",
                                                   parserArgs.inputPath / "training" / "segmentations",
                                                   parserArgs.size,
                                                   parserArgs.batchSize)

        validationDataGenerator = ModelDataGenerator(parserArgs.inputPath / "validation" / "images",
                                                     parserArgs.inputPath / "validation" / "segmentations",
                                                     parserArgs.size,
                                                     parserArgs.batchSize)

        # Run the trainer.
        trainer.Train(parserArgs.learningRate, parserArgs.patience, parserArgs.epochs,
                      trainingDataGenerator, validationDataGenerator, parserArgs.outputPath)
