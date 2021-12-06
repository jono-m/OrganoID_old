from commandline.Program import Program
import argparse
import pathlib


class Label(Program):
    def Name(self):
        return "label"

    def Description(self):
        return "Label single organoids in a detection image."

    def SetupParser(self, parser: argparse.ArgumentParser):
        parser.add_argument("detectionsPath", help="Path to detection images.", type=pathlib.Path)
        parser.add_argument("-O", dest="outputPath", default=None,
                            help="Path where labeled images will be saved.",
                            type=pathlib.Path)
        parser.add_argument("-A", dest="minArea", default=100,
                            type=int,
                            help="Remove organoids with an area smaller than a set number of pixels.")
        parser.add_argument("-T", dest="textSize", default=12,
                            type=int,
                            help="Draw text on RGB image with given font size.")
        parser.add_argument("--removeBorder", action="store_true", help="Remove organoids that are touching borders.")
        parser.add_argument("--rgb", action="store_true",
                            help="If set, an RGB version of each image will also be produced.")
        parser.add_argument("--edge", action="store_true",
                            help="If set, a version of each image with edge detection will also be produced.")
        parser.add_argument("--show", action="store_true", help="If set, the output images will be displayed.")

    def RunProgram(self, parserArgs: argparse.Namespace):
        from backend.ImageManager import LoadImages, SaveImage, ShowImage, LabelToRGB, SaveTIFFStack
        from backend.Label import Label, DetectEdges

        # Load detection images
        images = LoadImages(parserArgs.detectionsPath, size=[512, 512])

        count = 1
        for image in images:
            print("Labeling %d: %s" % (count, image.path))
            count += 1

            edges = image.DoOperation(DetectEdges)
            identified = image.DoOperation(lambda x: Label(x, parserArgs.minArea, parserArgs.removeBorder), True)

            outputImages = []
            if parserArgs.rgb:
                outputImages.append(("rgb", identified.DoOperation(lambda x: LabelToRGB(x, parserArgs.textSize), True)))
            if parserArgs.edge:
                outputImages.append(("edges", edges))
            outputImages.append(("labeled", identified))

            if parserArgs.show:
                [ShowImage(outputImage.frames[0]) for (_, outputImage) in outputImages]

            if parserArgs.outputPath is not None:
                for name, outputImage in outputImages:
                    extension = outputImage.path.suffix
                    fileName = outputImage.path.stem + "_" + name + extension
                    savePath = parserArgs.outputPath / fileName
                    print(savePath)
                    if len(outputImage.frames) > 1:
                        SaveTIFFStack(outputImage.frames, savePath)
                    else:
                        SaveImage(outputImage.frames[0], savePath)
