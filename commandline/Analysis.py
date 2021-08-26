from commandline.Program import Program
import argparse
import pathlib


class Analysis(Program):
    def Name(self):
        return "analysis"

    def Description(self):
        return "Analysis of segmented and labeled organoid images."

    def SetupParser(self, parser: argparse.ArgumentParser):
        parser.add_argument("imagesPath", help="Path to labeled images to process.", type=pathlib.Path)
        parser.add_argument("-O", dest="outputPath", default=None,
                            help="Path where results will be saved.",
                            type=pathlib.Path)
        parser.add_argument("--plot", action="store_true", help="Plot results.")
        parser.add_argument("--print", action="store_true", help="Print results to console.")

    def RunProgram(self, parserArgs: argparse.Namespace):
        from backend.ImageManager import LoadImages
        from backend.Analyzer import Analyzer
        import matplotlib.pyplot as plt
        import numpy as np
        images = LoadImages(parserArgs.imagesPath, size=[512, 512])

        count = 1
        analyzer = Analyzer()
        imageNames = []
        for image in images:
            print("Analyzing %d: %s" % (count, image.path))
            count += 1
            if len(image.frames) == 1:
                imageNames.append(image.path.name)
            else:
                imageNames += [image.path.name + "_" + str(i + 1) for i in range(len(image.frames))]
            [analyzer.AnalyzeImage(frame) for frame in image.frames]

        if parserArgs.outputPath:
            parserArgs.outputPath.mkdir(parents=True, exist_ok=True)
            with open(parserArgs.outputPath / (self.JobName() + ".csv"), 'w', newline='') as csvfile:
                csvfile.write(
                    "Image Name, Organoid Count, Total Area, Mean Area, Median Area, Area STD, Individual Areas\n")
                for (name, timePoint) in zip(imageNames, analyzer.timePoints):
                    csvfile.write("%s, %d, %d, %d, %d, %d, %s\n" %
                                  (name,
                                   len(timePoint.organoidAreas),
                                   np.sum(timePoint.organoidAreas),
                                   np.mean(timePoint.organoidAreas),
                                   np.median(timePoint.organoidAreas),
                                   np.std(timePoint.organoidAreas),
                                   ", ".join([str(x) for x in timePoint.organoidAreas])))

        if parserArgs.print:
            for (name, timePoint) in zip(imageNames, analyzer.timePoints):
                print("%s, %d, %d, %d, %d, %d, %s\n" %
                      (name,
                       len(timePoint.organoidAreas),
                       np.sum(timePoint.organoidAreas),
                       np.mean(timePoint.organoidAreas),
                       np.median(timePoint.organoidAreas),
                       np.std(timePoint.organoidAreas),
                       timePoint.organoidAreas))

        if parserArgs.plot:
            t = range(len(analyzer.timePoints))

            plt.subplot(1, 4, 1)
            plt.title("Organoid Count")
            plt.xlabel("Frame")
            plt.ylabel("Number")
            plt.plot(t, [len(point.organoidAreas) for point in analyzer.timePoints])

            plt.subplot(1, 4, 2)
            plt.title("Organoid Total Area")
            plt.xlabel("Frame")
            plt.ylabel("Area (Pixels)")
            plt.plot(t, [np.sum(point.organoidAreas) for point in analyzer.timePoints])

            plt.subplot(1, 4, 3)
            plt.title("Mean Area Per Organoid")
            plt.xlabel("Frame")
            plt.ylabel("Area (Pixels)")
            plt.errorbar(t, [np.mean(point.organoidAreas) for point in analyzer.timePoints],
                         [np.std(point.organoidAreas) for point in analyzer.timePoints])

            plt.subplot(1, 4, 4)
            plt.title("Median Area Per Organoid")
            plt.xlabel("Frame")
            plt.ylabel("Area (Pixels)")
            plt.errorbar(t, [np.median(point.organoidAreas) for point in analyzer.timePoints],
                         [np.std(point.organoidAreas) for point in analyzer.timePoints])

            plt.show()
