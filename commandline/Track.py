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
        parser.add_argument("-O", dest="outputPath", default=None,
                            help="Path where results will be saved.",
                            type=pathlib.Path)
        parser.add_argument("--raw", action="store_true", help="If set, all images will be saved as separate frames.")
        parser.add_argument("--gif", action="store_true", help="If set, all images will be saved as an animated GIF.")
        parser.add_argument("--show", action="store_true", help="Show frames.")
        parser.add_argument("--label", action="store_true", help="Draw text on images.")

    def RunProgram(self, parserArgs: argparse.Namespace):
        from backend.ImageManager import LoadImages, SaveImages, ShowImage, EmphasizeLabeled
        from backend.Tracker import Tracker
        images = LoadImages(parserArgs.imagesPath, size=[512, 512])

        count = 1

        tracker = Tracker()

        for image in images:
            print("Tracking %d: %s" % (count, image.path))
            count += 1
            tracker.Track(image.image)

        outputImages = [EmphasizeLabeled(tracked, (255, 255, 255), (0, 255, 0)) for tracked in
                        tracker.GetTrackedImages()]

        if parserArgs.show:
            ShowImage(outputImages)
        if parserArgs.outputPath is not None and parserArgs.gif:
            SaveImages(outputImages, parserArgs.outputPath / self.JobName() / "trackResults.gif")