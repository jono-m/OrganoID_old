from commandline.Program import Program
import argparse
import pathlib
import math


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
        parser.add_argument("-M", dest="grouping", default=None,
                            help="Optional matrix grouping of frames (e.g. every set of G=10 images is a separate column).",
                            type=int)
        parser.add_argument("--unmask", action="store_true", help="Also produce intensity sum for unmasked images.")
        parser.add_argument("--plot", action="store_true", help="Plot results.")
        parser.add_argument("--print", action="store_true", help="Print results to console.")

    def RunProgram(self, parserArgs: argparse.Namespace):
        from backend.ImageManager import LoadImages
        import numpy as np
        images = list(LoadImages(parserArgs.imagesPath, size=[512, 512]))
        masks = list(LoadImages(parserArgs.masksPath, size=[512, 512], mode="1"))

        intensities = []
        unmaskedIntensities = []
        imageNames = []
        for (mask, image) in zip(masks, images):
            count = len(image.frames)
            if count == 1:
                names = [image.path.stem]
            else:
                names = [image.path.stem + "_" + str(i + 1) for i in range(count)]
            imageNames += names
            for (name, maskFrame, imageFrame) in zip(names, mask.frames, image.frames):
                print("Masking: %s" % name)
                maskImage = np.where(maskFrame, imageFrame, 0)
                intensities.append(np.sum(maskImage))
                unmaskedIntensities.append(np.sum(imageFrame))

        def GetText(values):
            allText = ""
            if parserArgs.grouping is not None:
                imageCount = len(imageNames)
                groupCount = math.ceil(imageCount / parserArgs.grouping)
                allText += "N, " + ", ".join([str(i) for i in range(parserArgs.grouping)]) + "\n"
                for group in range(groupCount):
                    groupText = "Group " + str(group + 1)
                    groupStart = group * parserArgs.grouping
                    for i in range(groupStart, groupStart + parserArgs.grouping):
                        if i > len(values):
                            groupText += ", None"
                        else:
                            groupText += ", " + str(values[i])
                    allText += groupText + "\n"
            else:
                allText += "Name, Total Intensity\n"
                for t in range(len(values)):
                    allText += "%s, %d\n" % (imageNames[t], values[t])

            return allText

        if parserArgs.outputPath:
            parserArgs.outputPath.mkdir(parents=True, exist_ok=True)
            with open(parserArgs.outputPath / (self.JobName() + ".csv"), 'w', newline='') as csvfile:
                csvfile.write(GetText(intensities))
            if parserArgs.unmask:
                with open(parserArgs.outputPath / (self.JobName() + "_UNMASKED.csv"), 'w', newline='') as csvfile:
                    csvfile.write(GetText(unmaskedIntensities))

        if parserArgs.print:
            print(GetText(intensities))
            if parserArgs.unmask:
                print(GetText(unmaskedIntensities))

        if parserArgs.plot:
            import matplotlib.pyplot as plt

            if parserArgs.unmask:
                plt.subplot(1, 2, 1)
            plt.plot(range(len(intensities)), intensities)
            plt.title("Total Intensity")
            plt.xlabel("Frame")
            plt.ylabel("Intensity (a.u.)")
            if parserArgs.unmask:
                plt.subplot(1, 2, 2)
                plt.plot(range(len(unmaskedIntensities)), unmaskedIntensities)
                plt.title("Total Intensity (UNMASKED)")
                plt.xlabel("Frame")
                plt.ylabel("Intensity (a.u.)")

            plt.show()
