from commandline.Program import Program
import argparse
import pathlib


class TimeSeriesAnalysis(Program):
    def Name(self):
        return "tsa"

    def Description(self):
        return "Time-series analysis of segmented and labeled organoid images."

    def SetupParser(self, parser: argparse.ArgumentParser):
        parser.add_argument("imagesPath", help="Path to labeled images to process.", type=pathlib.Path)
        parser.add_argument("outputPath", help="Path where results will be saved.", type=pathlib.Path)

    def RunProgram(self, parserArgs: argparse.Namespace):
        from backend.ImageManager import LoadImages
        from backend.TimeSeriesAnalyzer import TimeSeriesAnalyzer
        import numpy as np
        images = LoadImages(parserArgs.imagesPath, size=[512, 512])

        count = 1
        analyzer = TimeSeriesAnalyzer()
        for image in images:
            print("Analyzing %d: %s" % (count, image.path))
            count += 1
            [analyzer.AnalyzeImage(frame) for frame in image.frames]

        parserArgs.outputPath.mkdir(parents=True, exist_ok=True)
        with open(parserArgs.outputPath / (self.JobName() + ".csv"), 'w', newline='') as csvfile:
            csvfile.write("Time Point, Organoid Count, Mean Area, Median Area, Area STD, Individual Areas\n")
            for t in range(len(analyzer.timePoints)):
                point = analyzer.timePoints[t]
                csvfile.write("%d, %d, %d, %d, %d, %s\n" %
                              (t,
                               len(point.organoidAreas),
                               np.mean(point.organoidAreas),
                               np.median(point.organoidAreas),
                               np.std(point.organoidAreas),
                               point.organoidAreas))
