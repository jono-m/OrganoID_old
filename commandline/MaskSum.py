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
        images = LoadImages(parserArgs.imagesPath, size=[512, 512], mode="L")
        masks = LoadImages(parserArgs.masksPath, size=[512, 512], mode="1")

        count = 1

        imageFrames = [frame for image in images for frame in image.frames]
        maskFrames = [frame for mask in masks for frame in mask.frames]
        intensities = []
        for (mask, image) in zip(maskFrames, imageFrames):
            print("Masking %d" % count)
            count += 1
            maskImage = np.where(mask, image, 0)
            intensities.append(np.sum(maskImage))

        if parserArgs.outputPath:
            parserArgs.outputPath.mkdir(parents=True, exist_ok=True)
            with open(parserArgs.outputPath / (self.JobName() + ".csv"), 'w', newline='') as csvfile:
                csvfile.write(
                    "Time Point, Total Intensity\n")
                for t in range(len(intensities)):
                    csvfile.write("%d, %d\n" % (t, intensities[t]))

        if parserArgs.print:
            for intensity in intensities:
                print(intensity)

        if parserArgs.plot:
            import matplotlib.pyplot as plt

            plt.plot(range(len(intensities)), intensities)
            plt.title("Total Intensity")
            plt.xlabel("Frame")
            plt.ylabel("Intensity (a.u.)")
            plt.show()