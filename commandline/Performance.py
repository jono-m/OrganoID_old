import numpy as np
from commandline.Program import Program
import argparse
import pathlib


class Performance(Program):
    def Name(self):
        return "performance"

    def Description(self):
        return "Measure statistics for image files."

    def SetupParser(self, parser: argparse.ArgumentParser):
        parser.add_argument("modelPath", help="Path to trained OrganoID model", type=pathlib.Path)
        parser.add_argument("inputPath",
                            help="Path to images to analyze. With subdirectories images/ and segmentations/",
                            type=pathlib.Path)
        parser.add_argument("-T", dest="threshold", default=0.5, type=float,
                            help="Set a fixed threshold [THRESHOLD] for segmentation (0-1).")
        parser.add_argument("-S", dest="sweep", type=int, default=0,
                            help="Sweep the threshold over [SWEEP] points.")
        parser.add_argument("--iou", action="store_true", help="Compute intersection-over-union (Jaccard index).")
        parser.add_argument("--auc", action="store_true",
                            help="Compute the area under the ROC curve. Only valid for sweeps.")
        parser.add_argument("--plot", action="store_true", help="Plot useful info.")

    def RunProgram(self, parserArgs: argparse.Namespace):
        from sklearn.metrics import auc
        from backend.Segmenter import Segmenter
        from backend.ImageManager import LoadImages, ContrastOp
        import matplotlib.pyplot as plt
        images = LoadImages(parserArgs.inputPath / "images", (512, 512), mode="L")
        segmentations = LoadImages(parserArgs.inputPath / "segmentations", (512, 512), mode="1")

        segmenter = Segmenter(parserArgs.modelPath)

        cmats_by_threshold = {}
        count = 1
        for image, segmentation in zip(images, segmentations):
            self.printRep("Segmenting image %d: %s" % (count, image.path.name))
            count += 1
            predicted = image.DoOperation(ContrastOp).DoOperation(segmenter.Segment)

            if parserArgs.sweep:
                sweeps = np.linspace(0, 1, parserArgs.sweep)
            else:
                sweeps = [parserArgs.threshold]

            for threshold in sweeps:
                if threshold not in cmats_by_threshold:
                    cmats_by_threshold[threshold] = []
                cmats = [confusionMatrix(segmentationFrame, predictedFrame > threshold) for
                         segmentationFrame, predictedFrame in zip(segmentation.frames, predicted.frames)]
                cmats_by_threshold[threshold] += cmats

        self.printRep()
        print("Done.")

        if parserArgs.iou:
            ious_by_threshold = {threshold: [IOU(cmat) for cmat in cmats_by_threshold[threshold]] for threshold in
                                 cmats_by_threshold}

            for threshold in ious_by_threshold:
                ious = ious_by_threshold[threshold]
                print("IOU (T=%f): %f\u00B1%f" % (threshold, np.mean(ious), np.std(ious)))

            if parserArgs.plot:
                [plt.hist(ious_by_threshold[threshold], bins="auto") for threshold in ious_by_threshold]
                plt.show()

        if parserArgs.auc:
            tprs = [np.mean([TPR(cmat) for cmat in cmats_by_threshold[threshold]]) for threshold in cmats_by_threshold]
            fprs = [np.mean([FPR(cmat) for cmat in cmats_by_threshold[threshold]]) for threshold in cmats_by_threshold]

            areaUnderCurve = auc(fprs, tprs)

            if parserArgs.plot:
                plt.plot(fprs, tprs)
                plt.xlim([0, 1])
                plt.ylim([0, 1])
                plt.xlabel("False Positive Rate")
                plt.ylabel("True Positive Rate")
                plt.title("ROC (AUC: %f)" % areaUnderCurve)
                plt.show()
            print("AUC: %f" % areaUnderCurve)
        print("Done.")


def IOU(cmat):
    (TP, FP, FN, TN) = cmat
    return TP / (FP + FN + TP)


def TPR(cmat):
    (TP, FP, FN, TN) = cmat
    return TP / (TP + FN)


def FPR(cmat):
    (TP, FP, FN, TN) = cmat
    return FP / (FP + TN)


def confusionMatrix(true: np.ndarray, predicted: np.ndarray):
    true = true.astype(dtype=bool)
    predicted = predicted.astype(dtype=bool)
    TP = np.count_nonzero(np.bitwise_and(true == predicted, true))
    FP = np.count_nonzero(np.bitwise_and(true != predicted, np.bitwise_not(true)))
    FN = np.count_nonzero(np.bitwise_and(true != predicted, true))
    TN = np.count_nonzero(np.bitwise_and(true == predicted, np.bitwise_not(true)))
    return TP, FP, FN, TN
