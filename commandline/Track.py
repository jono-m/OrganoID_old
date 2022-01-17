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
        parser.add_argument("outputPath", help="Directory where results will be saved.", type=pathlib.Path)
        parser.add_argument("-measure", nargs="+", dest="features", help="List of features to measure. "
                                                                         "See https://scikit-image.org/docs/dev/api/skimage.measure.html#skimage.measure.regionprops for available features")
        parser.add_argument("--individual", action="store_true",
                            help="If set, tracked images will be saved as separate frames.")
        parser.add_argument("--gif", action="store_true",
                            help="If set, tracked images will be saved as a GIF video.")
        parser.add_argument("--batch", action="store_true",
                            help="If set, each image will be treated as a separate tracking stack.")

    def RunProgram(self, parserArgs: argparse.Namespace):
        from backend.ImageManager import LoadImages, SaveGIF, LabelTracks, SaveImage
        from backend.Tracker import Tracker

        if parserArgs.batch:
            # Load labeled images
            labeledImages = LoadImages(parserArgs.labeledImagesPath, size=[512, 512])
            originalImages = LoadImages(parserArgs.originalImagesPath, size=[512, 512], mode='L')

            count = 1

            # Track images
            for labeledImage, originalImage in zip(labeledImages, originalImages):
                # Load tracker
                tracker = Tracker()
                for i, frame in enumerate(labeledImage.frames):
                    tracker.Track(frame)
                    print("Tracking %d: %s (%d)" % (count, labeledImage.path, i))
                    count += 1

                # Load original images
                originalFrames = [frame * parserArgs.brightness for frame in originalImage.frames]
                outputImages = LabelTracks(tracker.GetTracks(), (255, 255, 255, 255), 255, 50, (0, 205, 108), {},
                                           originalFrames)

                if parserArgs.gif:
                    # Export GIF
                    SaveGIF(outputImages, parserArgs.outputPath / (originalImage.path.stem + "_tracked.gif"))

                if parserArgs.features:
                    trackInfo = "Frame, Original Label, Organoid ID, " + ", ".join(parserArgs.features)
                    for frameNumber in range(count - 1):
                        tracks = [track for track in tracker.GetTracks() if track.WasDetected(frameNumber)]
                        for track in tracks:
                            data = track.Data(frameNumber)
                            rp = data.GetRP()
                            trackInfo += "%d, %d, %d, %s\n" % (frameNumber, rp.label, track.id,
                                                               ", ".join([str(eval("rp." + feature)) for feature in
                                                                          parserArgs.features]))

                    # Export track data
                    csvFile = open(parserArgs.outputPath / (originalImage.path.stem + "_trackResults.csv"), 'w+')
                    csvFile.write(trackInfo)
                    csvFile.close()
            return

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
        originalFrames = [frame for baseImage in originalImages for frame in baseImage.frames]
        outputImages = LabelTracks(tracker.GetTracks(), (255, 255, 255, 255), 255, 50, (0, 205, 108), {},
                                   originalFrames)

        if parserArgs.gif:
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

        if parserArgs.features:
            trackInfo = "Frame, Original Label, Organoid ID, " + ", ".join(parserArgs.features) + "\n"
            for frameNumber in range(count - 1):
                tracks = [track for track in tracker.GetTracks() if track.WasDetected(frameNumber)]
                for track in tracks:
                    data = track.Data(frameNumber)
                    rp = data.GetRP()
                    trackInfo += "%d, %d, %d, %s\n" % (frameNumber, rp.label, track.id,
                                                       ", ".join([str(eval("rp." + feature, {}, {"rp": rp})) for feature in
                                                                  parserArgs.features]))

            # Export track data
            csvFile = open(parserArgs.outputPath / "trackResults.csv", 'w+')
            csvFile.write(trackInfo)
            csvFile.close()
