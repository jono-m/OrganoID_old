# Track.py -- sub-program to track organoids in a series of labeled images.

from commandline.Program import Program
import argparse
import pathlib


class Track(Program):
    def Name(self):
        return "track"

    def Description(self):
        return "Track organoids over time in a sequence of labeled images."

    def SetupParser(self, parser: argparse.ArgumentParser):
        parser.add_argument("labeledImagesPath", help="Path to labeled images to process.", type=pathlib.Path)
        parser.add_argument("originalImagesPath", help="Path to original microscopy images for overlaying.",
                            type=pathlib.Path)
        parser.add_argument("outputPath", help="Path where results will be saved.", type=pathlib.Path)
        parser.add_argument("-B", dest="brightness", default=1, help="Brightness multiplier for original image.",
                            type=float)
        parser.add_argument("--individual", action="store_true",
                            help="If set, tracked images will be saved as separate frames.")

    def RunProgram(self, parserArgs: argparse.Namespace):
        from backend.ImageManager import LoadImages, SaveGIF, ShowImage, LabelTracks, SaveImage
        from backend.Tracker import Tracker

        # Load labeled images
        labeledImages = LoadImages(parserArgs.labeledImagesPath, size=[512, 512])

        count = 1

        # Load tracker
        tracker = Tracker()

        # Track images
        for image in labeledImages:
            for i, frame in enumerate(image.frames):
                tracker.Track(frame)
                print("Tracking %d: %s (%d)" % (count, image.path, i))
                count += 1

        # Load original images
        originalImages = LoadImages(parserArgs.originalImagesPath, size=[512, 512], mode='L')
        originalFrames = [frame * parserArgs.brightness for baseImage in originalImages for frame in baseImage.frames]
        outputImages = LabelTracks(tracker.GetTracks(), (255, 255, 255, 255), 255, 50, (0, 205, 108), (255, 0, 0),
                                   originalFrames)

        # Export GIF
        SaveGIF(outputImages, parserArgs.outputPath / "trackResults.gif")

        # Export original images
        if parserArgs.individual:
            i = 0
            for outputImage in outputImages:
                fileName = "trackResults_" + str(i) + ".png"
                i += 1
                savePath = parserArgs.outputPath / fileName
                SaveImage(outputImage, savePath)

        # Export track data
        csvFile = open(parserArgs.outputPath / "trackResults.csv", 'w+')
        csvFile.write("Frame, Original Label, Organoid ID\n")
        for frameNumber in range(count - 1):
            for track in tracker.GetTracks():
                data = track.DataAtFrame(frameNumber)
                if data is None or not data.wasDetected:
                    continue
                else:
                    csvFile.write("%d, %d, %d\n" % (frameNumber, data.label, track.id))
        csvFile.close()
