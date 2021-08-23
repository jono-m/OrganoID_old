from commandline.Program import Program
import argparse
import pathlib


class MaskSum(Program):
    def Name(self):
        return "masksum"

    def Description(self):
        return "Applies segmentation masks to images and computes an intensity sum."

    def SetupParser(self, parser: argparse.ArgumentParser):
        parser.add_argument("imagesPath", help="Path to image(s) to analyze.", type=pathlib.Path)
        parser.add_argument("masksPath", help="Path to image mask(s).", type=pathlib.Path)
        parser.add_argument("-O", dest="outputPath", default=None,
                            help="Path where results will be saved.",
                            type=pathlib.Path)
        parser.add_argument("--plot", action="store_true", help="Plot results.")
        parser.add_argument("--print", action="store_true", help="Print results to console.")

    def RunProgram(self, parserArgs: argparse.Namespace):
        from backend.ImageManager import LoadImages
        import numpy as np
        images = list(LoadImages(parserArgs.imagesPath, size=[512, 512], mode="L"))
        masks = list(LoadImages(parserArgs.masksPath, size=[512, 512], mode="1"))

        imageFrames = [frame for image in images for frame in image.frames]
        imageNames = []
        for image in images:
            count = len(image.frames)
            if count == 1:
                imageNames.append(image.path.name)
            else:
                imageNames += [image.path.name + "_" + str(i+1) for i in range(count)]

        maskFrames = [frame for mask in masks for frame in mask.frames]
        intensities = []
        count = 1

        for (name, mask, image) in zip(imageNames, maskFrames, imageFrames):
            print("Masking %d: %s" % (count, name))
            count += 1
            maskImage = np.where(mask, image, 0)
            intensities.append(np.sum(maskImage))

        if parserArgs.outputPath:
            parserArgs.outputPath.mkdir(parents=True, exist_ok=True)
            with open(parserArgs.outputPath / (self.JobName() + ".csv"), 'w', newline='') as csvfile:
                csvfile.write(
                    "Name, Total Intensity\n")
                for t in range(len(intensities)):
                    csvfile.write("%s, %d\n" % (imageNames[t], intensities[t]))

        if parserArgs.print:
            for (name, intensity) in zip(imageNames, intensities):
                print("%s: %d" % (name, intensity))

        if parserArgs.plot:
            import matplotlib.pyplot as plt

            plt.plot(range(len(intensities)), intensities)
            plt.title("Total Intensity")
            plt.xlabel("Frame")
            plt.ylabel("Intensity (a.u.)")
            plt.show()