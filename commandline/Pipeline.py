from commandline.Program import Program
import argparse
import pathlib


class Pipeline(Program):
    def Name(self):
        return "pipeline"

    def Description(self):
        return "Run the full pipeline."

    def SetupParser(self, parser: argparse.ArgumentParser):
        parser.add_argument("modelPath", help="Path to trained OrganoID model", type=pathlib.Path)
        parser.add_argument("imagesPath", help="Path to raw images to process.", type=pathlib.Path)
        parser.add_argument("-O", dest="outputPath", default=None,
                            help="Path where processed images will be saved.",
                            type=pathlib.Path)
        parser.add_argument("-T", dest="threshold", default=0.5,
                            type=float,
                            help="Segmentation threshold (0-1).")
        parser.add_argument("-W", dest="watershedThresh", default=None,
                            type=float,
                            help="Watershed threshold (0-1).")
        parser.add_argument("-A", dest="minArea", default=20,
                            type=int,
                            help="Area cutoff.")
        parser.add_argument("-B", dest="borderCutoff", default=None,
                            type=float,
                            help="Remove organoids that are touching borders with more than [BORDERCUTOFF]*Diameter "
                                 "pixels (0-1).")
        parser.add_argument("--thresh", action="store_true",
                            help="If set, the output images will also be produced in raw "
                                 "format, which is needed for postprocessing.")
        parser.add_argument("--heat", action="store_true", help="If set, the output images will also be produced in a "
                                                                "heatmap format, which is good for visualizing raw "
                                                                "network behavior.")
        parser.add_argument("--edge", action="store_true", help="If set, the output images will also be produced in a "
                                                                "heatmap format, which is good for visualizing raw "
                                                                "network behavior.")
        parser.add_argument("--rgb", action="store_true", help="If set, an RGB version of each image will be produced.")
        parser.add_argument("--show", action="store_true", help="If set, the output images will be displayed.")

    def RunProgram(self, parserArgs: argparse.Namespace):
        from backend.ImageManager import LoadImages, SaveImage, ShowImage, LabelToRGB, SaveTIFFStack, ContrastOp
        from backend.Label import Label, Edges
        from backend.Segmenter import Segmenter
        from backend.PostProcessing import PostProcess
        import numpy as np
        import matplotlib.pyplot as plt
        images = LoadImages(parserArgs.imagesPath, size=[512, 512], mode="L")
        segmenter = Segmenter(parserArgs.modelPath)

        count = 1
        for image in images:
            print("Segenting %d: %s" % (count, image.path))
            print("Processing %d: %s" % (count, image.path))
            count += 1

            segmented_raw = image.DoOperation(ContrastOp).DoOperation(segmenter.Segment)

            labeled = segmented_raw.DoOperation(
                lambda x: Label(x, parserArgs.threshold, parserArgs.watershedThresh))

            edges = segmented_raw.DoOperation(
                lambda x: Edges(x, parserArgs.threshold, parserArgs.watershedThresh))

            postProcessed = labeled.DoOperation(
                lambda x: PostProcess(x, parserArgs.minArea, parserArgs.borderCutoff, True))

            outputImages = []
            if parserArgs.heat:
                outputImages.append(
                    ("heat",
                     segmented_raw.DoOperation(lambda x: (plt.get_cmap("hot")(x)[:, :, :3] * 255).astype(np.uint8))))
            if parserArgs.edge:
                outputImages.append(("edge", edges.DoOperation(lambda x: x * 2048)))
            if parserArgs.thresh:
                outputImages.append(("threshold", segmented_raw.DoOperation(lambda x: x > parserArgs.threshold)))
            if parserArgs.rgb:
                outputImages.append(("rgb", postProcessed.DoOperation(LabelToRGB)))

            if parserArgs.show:
                [ShowImage(outputImage.frames[0]) for (_, outputImage) in outputImages]

            if parserArgs.outputPath is not None:
                for name, outputImage in outputImages:
                    extension = outputImage.path.suffix
                    fileName = outputImage.path.stem + "_" + name + extension
                    savePath = parserArgs.outputPath / self.JobName() / fileName
                    print(savePath)
                    if len(outputImage.frames) > 1:
                        SaveTIFFStack(outputImage.frames, savePath)
                    else:
                        SaveImage(outputImage.frames[0], savePath)