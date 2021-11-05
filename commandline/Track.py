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
        parser.add_argument("-D", dest="deleteAfter", default=-1,
                            help="Discard a track when it has been missing for a given number of frames.",
                            type=float)
        parser.add_argument("-CM", dest="missingCost", default=10,
                            help="Cost for considering an organoid as 'lost' for a frame, instead of assigning it to an"
                                 " existing organoid track. A higher value assumes that organoids are rarely lost in "
                                 "sequential images. A lower value allows for more forgiveness. Values are "
                                 "in pixels, as relative to the distance an organoid might move over one frame.")
        parser.add_argument("-CN", dest="newCost", default=200,
                            help="Cost for considering an organoid as 'new' for a frame, instead of assigning it to an"
                                 " existing organoid track. A higher value assumes that new organoids rarely "
                                 "appear in sequential images (after the first image). "
                                 "A lower value will allow for more organoid tracks over time. Values are "
                                 "in pixels, as relative to the distance an organoid might move over one frame.")
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
        tracker.costOfMissingOrganoid = parserArgs.missingCost
        tracker.costOfNewOrganoid = parserArgs.newCost
        tracker.deleteTracksAfterMissing = parserArgs.deleteAfter

        # Track images
        for image in labeledImages:
            print("Tracking %d: %s" % (count, image.path))
            count += 1
            [tracker.Track(frame) for frame in image.frames]

        # Load original images
        originalImages = LoadImages(parserArgs.originalImagesPath, size=[512, 512], mode='L')
        originalFrames = [frame * parserArgs.brightness for baseImage in originalImages for frame in baseImage.frames]
        outputImages = LabelTracks(tracker.GetTracks(), (255, 255, 255, 255), 255, 50, (0, 255, 0), (255, 0, 0),
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
        with open(parserArgs.outputPath / "trackResults.csv", 'w+', newline='') as csvfile:
            csvfile.write(
                "Organoid ID, " + ", ".join([("Area(t=%d)" % i) for i in range(len(outputImages))]) + "\n")
            for track in tracker.GetTracks():
                areas = ["" for i in range(track.firstFrame)]

                for data in track.data:
                    if data.wasDetected:
                        areas.append(str(data.area))
                    else:
                        areas.append("Missing")
                line = str(track.id) + ", " + ", ".join(areas) + "\n"
                csvfile.write(line)
