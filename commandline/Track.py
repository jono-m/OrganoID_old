from commandline.Program import Program
import argparse
import pathlib


class Track(Program):
    def Name(self):
        return "track"

    def Description(self):
        return "Track organoids over time in a sequence of images."

    def SetupParser(self, parser: argparse.ArgumentParser):
        parser.add_argument("imagesPath", help="Path to labeled images to process.", type=pathlib.Path)
        parser.add_argument("overlayPath", help="Path to images for overlaying.", type=pathlib.Path)
        parser.add_argument("-O", dest="outputPath", default=None,
                            help="Path where results will be saved.",
                            type=pathlib.Path)
        parser.add_argument("--raw", action="store_true", help="If set, all images will be saved as separate frames.")
        parser.add_argument("--gif", action="store_true", help="If set, all images will be saved as an animated GIF.")
        parser.add_argument("--analyze", action="store_true", help="If set, organoids will also be analyzed.")
        parser.add_argument("--show", action="store_true", help="Show frames.")
        parser.add_argument("--label", action="store_true", help="Draw text on images.")

    def RunProgram(self, parserArgs: argparse.Namespace):
        from backend.ImageManager import LoadImages, SaveGIF, ShowImage, LabelTracks, SaveImage
        from backend.Tracker import Tracker
        images = LoadImages(parserArgs.imagesPath, size=[512, 512])

        count = 1

        tracker = Tracker()

        for image in images:
            print("Tracking %d: %s" % (count, image.path))
            count += 1
            [tracker.Track(frame) for frame in image.frames]

        baseImages = LoadImages(parserArgs.overlayPath, size=[512, 512], mode='L')
        baseFrames = [frame for baseImage in baseImages for frame in baseImage.frames]
        outputImages = LabelTracks(tracker.GetTracks(), (255, 255, 255, 255), 255, 50, (0, 255, 0), (255, 0, 0),
                                   baseFrames)

        if parserArgs.show:
            [ShowImage(image) for image in outputImages]
        if parserArgs.outputPath is not None:
            if parserArgs.gif:
                SaveGIF(outputImages, parserArgs.outputPath / self.JobName() / "trackResults.gif")

            if parserArgs.raw:
                i = 0
                for outputImage in outputImages:
                    fileName = "trackResults_" + str(i) + ".png"
                    i += 1
                    savePath = parserArgs.outputPath / self.JobName() / fileName
                    SaveImage(outputImage.frames[0], savePath)

            if parserArgs.analyze:
                with open(parserArgs.outputPath / self.JobName() / (self.JobName() + ".csv"), 'w',
                          newline='') as csvfile:
                    csvfile.write(
                        "Organoid ID, " + ", ".join([("Area(t=%d)" % i) for i in range(len(outputImages))]) + "\n")
                    for track in tracker.GetTracks():
                        areas = []
                        for data in track.data:
                            if data.detection:
                                areas.append(str(data.area))
                            else:
                                areas.append("None")
                        line = str(track.id) + ", " + ", ".join(areas) + "\n"
                        csvfile.write(line)
