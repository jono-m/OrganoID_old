from PIL import Image, ImageQt, ImageOps
from typing import Optional
from PySide6.QtWidgets import QFrame, QLabel, QPushButton, QFileDialog, QGridLayout, QVBoxLayout, QHBoxLayout, \
    QSizePolicy, QSlider, QCheckBox, QSpinBox, QWidget, QDoubleSpinBox
from PySide6.QtGui import QPixmap, Qt, QPainter, QFont, QFontMetrics
from PySide6.QtCore import QPoint, QSize
from skimage.measure import regionprops
from skimage.color import label2rgb
import numpy as np
from pathlib import Path
from training.Segmenter import Segmenter, PostSegment
from training.watershed import Watershed, PostProcess
import matplotlib.pyplot as plt

from time import time


class PipelineWidget(QFrame):
    def __init__(self):
        super().__init__()

        self._interpreter = Segmenter(Path(r"assets\model.tflite"))

        self._fileLabel = QLabel("Browse for file...")
        self._browseButton = QPushButton("Browse")
        self._browseButton.clicked.connect(self.BrowseForFile)

        layoutA = QVBoxLayout()
        layoutA.setAlignment(Qt.AlignTop)
        layoutB = QHBoxLayout()
        layoutB.setAlignment(Qt.AlignLeft)
        layoutC = QGridLayout()

        layoutB.addWidget(self._fileLabel)
        layoutB.addWidget(self._browseButton)
        layoutA.addLayout(layoutB)

        self._originalImageWidget = ImageWidget("Original")
        self._segmentedImageWidget = ImageWidget("NN Output")
        self._postSegmentWidget = ImageWidget("Threshold Segmentation")
        self._minimaImageWidget = ImageWidget("Watershed Initializers")
        self._watershedImageWidget = ImageWidget("Watershed Separation")
        self._finalImageWidget = ImageWidget("Post Processed")

        self._thresholdSlider = LabeledSlider("Threshold", self.UpdateSegmentationPost, 50)
        self._dustSegmentation = LabeledSpinBox("Morphological Opening (Dust Removal)", self.UpdateSegmentationPost, 1)

        self._centerThreshold = LabeledSpinBox("Threshold", self.UpdateWatershedRaw, 0.9, True)
        self._dustWatershed = LabeledSpinBox("Morphological Opening (Dust Removal)", self.UpdateWatershedRaw, 1)
        self._watershedLabelsCheckbox = Checkbox("Show Labels", self.UpdateWatershedPost, True)

        self._fgFinalSlider = LabeledSlider("Opacity", self.UpdateFinalPost, 20)
        self._finalLabelsCheckbox = Checkbox("Show Labels", self.UpdateFinalPost, True)

        layoutC.addWidget(self._originalImageWidget, 0, 0)

        layoutC.addWidget(self._segmentedImageWidget, 0, 1)

        layoutC.addWidget(self._postSegmentWidget, 0, 2)
        layoutC.addWidget(self._thresholdSlider, 1, 2)
        layoutC.addWidget(self._dustSegmentation, 2, 2)

        layoutC.addWidget(self._minimaImageWidget, 0, 3)
        layoutC.addWidget(self._centerThreshold, 1, 3)
        layoutC.addWidget(self._dustWatershed, 2, 3)

        layoutC.addWidget(self._watershedImageWidget, 0, 4)

        self.cullSlider = LabeledSlider("Border Removal", self.UpdateFinalRaw, 0.75)
        self.smallObjectsSpinBox = LabeledSpinBox("Area Cutoff", self.UpdateFinalRaw, 100)

        layoutC.addWidget(self._finalImageWidget, 0, 5)
        layoutC.addWidget(self.cullSlider, 1, 5)
        layoutC.addWidget(self.smallObjectsSpinBox, 2, 5)
        layoutC.addWidget(self._fgFinalSlider, 3, 5)

        layoutA.addLayout(layoutC)
        layoutA.addStretch(1)

        self._originalImageRaw = None
        self._originalImagePost = None
        self._segmentedImageRaw = None
        self._segmentedImagePost = None
        self._dtImage = None
        self._minimaImage = None
        self._watershedImageRaw = None
        self._watershedImagePost = None
        self._finalImageRaw = None
        self._finalImagePost = None

        self._originalSize = None

        self.setLayout(layoutA)

        self.SetAllVisible(False)

        self.OpenFile(
            r"C:\Users\jonoj\Documents\ML\AugmentedData\OrganoID_augment_2021_06_29_23_46_50\raw\testing\images\XY081.png")

    def SetAllVisible(self, visible):
        for a in self.children():
            if isinstance(a, QWidget) and a != self._browseButton and a != self._fileLabel:
                a.setVisible(visible)

    def BrowseForFile(self):
        filename, _ = QFileDialog.getOpenFileName(self, "Browse for Image")
        if filename:
            self.OpenFile(filename)

    def OpenFile(self, filename):
        self._fileLabel.setText(filename)
        self.UpdateOriginalRaw(Image.open(filename))

    def UpdateOriginalRaw(self, original: Image):
        if not self._originalImageWidget.isVisible():
            self.SetAllVisible(True)

        self._originalSize = original.size
        if original.mode == 'I' or original.mode == 'I;16':
            original = original.point(lambda x: x * (1 / 255)).convert(mode='L')
        self._originalImageRaw = np.array(original)
        self.UpdateOriginalPost()

    def UpdateOriginalPost(self):
        self._originalImagePost = self._originalImageRaw
        self._originalImageWidget.SetImage(self._originalImagePost, self._originalImageRaw)
        self.UpdateSegmentationRaw()

    def UpdateSegmentationRaw(self):
        image = np.array(
            Image.fromarray(self._originalImagePost).resize(self._interpreter.inputShape).convert(mode="L"))
        self._segmentedImageRaw = self._interpreter.Predict(image)
        self._segmentedImageWidget.SetImage(
            (plt.get_cmap("hot")(self._segmentedImageRaw)[:, :, :3] * 255).astype(np.uint8),
            self._originalImageRaw)
        self.UpdateSegmentationPost()

    def UpdateSegmentationPost(self):
        self._segmentedImagePost = PostSegment(self._segmentedImageRaw, self._thresholdSlider.value(),
                                               self._dustSegmentation.value())

        self._postSegmentWidget.SetImage(self._segmentedImagePost, self._originalImageRaw)
        self.UpdateWatershedRaw()

    def UpdateWatershedRaw(self):
        self._dtImage, self._minimaImage, self._watershedImageRaw = Watershed(self._segmentedImagePost,
                                                                              self._segmentedImageRaw,
                                                                              self._centerThreshold.value(),
                                                                              self._dustWatershed.value())
        self.UpdateWatershedPost()

    def UpdateWatershedPost(self):
        self._watershedImagePost = self._watershedImageRaw
        self._minimaImageWidget.SetImage(self._minimaImage > 0, self._originalImageRaw)
        self._watershedImageWidget.SetImage(self._watershedImagePost, self._originalImageRaw, label=True)
        self.UpdateFinalRaw()

    def UpdateFinalRaw(self):
        self._finalImageRaw = PostProcess(self._watershedImageRaw, self.smallObjectsSpinBox.value(),
                                          self.cullSlider.value())
        self.UpdateFinalPost()

    def UpdateFinalPost(self):
        self._finalImagePost = self._finalImageRaw
        self._finalImageWidget.SetImage(self._finalImagePost, self._originalImageRaw,
                                        imageAlpha=self._fgFinalSlider.value(), label=True)


class ImageWidget(QWidget):
    def __init__(self, title):
        super().__init__()
        self._title = title
        layout = QVBoxLayout()
        self._imageLabel = QLabel()
        self._imageLabel.setAlignment(Qt.AlignCenter)
        layout.addWidget(self._imageLabel)
        title = QLabel("<b>" + title + "</b>")
        title.setStyleSheet("""background-color: rgb(200, 200, 200)""")
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)
        self._saveButton = QPushButton("Save Image...")
        layout.addWidget(self._saveButton)
        self._saveButton.clicked.connect(self.SaveRaw)
        self.setLayout(layout)

        self._labels = []

        self._rawImage: Optional[QPixmap] = None
        self._imageLabel.setMinimumSize(256, 256)
        self._imageLabel.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.SetImage(None)

    def SaveRaw(self):
        if self._rawImage:
            filepath, _ = QFileDialog.getSaveFileName(self, "Save To...", filter="Image File (*.png *.jpg *.tif)")
            if filepath:
                self.pixmapWithLabels(self._rawImage.size()).save(filepath)

    def SetImage(self, image: np.ndarray = None, originalImage: np.ndarray = None, imageAlpha=1, label=False,
                 autoContrast=False):
        if image is None:
            preparedImage = Image.fromarray(np.zeros((512, 512))).convert(mode="L")
        else:
            height, width = originalImage.shape[:2]
            preparedImage = Image.fromarray(image).resize((width, height), resample=Image.NEAREST)
            if autoContrast:
                preparedImage = ImageOps.autocontrast(preparedImage)
            self._labels = []
            if label:
                originalImage = np.array(Image.fromarray(originalImage).convert(mode="RGB"))
                preparedImage = label2rgb(np.array(preparedImage), originalImage, bg_label=0, alpha=imageAlpha,
                                          image_alpha=1)
                preparedImage = Image.fromarray((preparedImage * 255).astype(np.uint8))
                props = regionprops(image)
                for prop in props:
                    y, x = prop.centroid
                    oldHeight, oldWidth = image.shape
                    x, y = self.Transform((x, y), QSize(width, height), QSize(oldWidth, oldHeight))
                    self._labels.append(((x, y), str(prop.label)))

        qt = ImageQt.ImageQt(preparedImage)
        self._rawImage = QPixmap.fromImage(qt)
        self.doResize()

    def resizeEvent(self, event) -> None:
        self.doResize()

    def doResize(self):
        self._imageLabel.setPixmap(self.pixmapWithLabels(self.size()))

    def pixmapWithLabels(self, size: QSize):
        pixmap = self._rawImage.scaled(size, Qt.KeepAspectRatio)
        painter = QPainter()

        painter.begin(pixmap)
        font = QFont("Arial", 12, QFont.Bold)
        metric = QFontMetrics(font)
        painter.setFont(font)
        painter.setPen(Qt.white)
        for (pt, text) in self._labels:
            x, y = self.Transform(pt, pixmap.size(), self._rawImage.size())
            rect = metric.tightBoundingRect(text)
            rect.moveCenter(QPoint(x, y))
            painter.drawText(rect, Qt.AlignCenter, text)

        painter.end()

        return pixmap

    @staticmethod
    def Transform(pt, newSize, oldSize):
        x, y = pt

        x = x / oldSize.width() * newSize.width()
        y = y / oldSize.height() * newSize.height()

        return x, y


class LabeledSlider(QWidget):
    def __init__(self, text, delegate, initialValue, tracking=False):
        super().__init__()
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMinimum(0)
        self.slider.setMaximum(100)
        self.slider.setValue(initialValue)
        self.slider.setTracking(tracking)
        self.slider.valueChanged.connect(delegate)
        thresholdLayout = QHBoxLayout()
        thresholdLayout.addWidget(QLabel(text))
        thresholdLayout.addWidget(self.slider)
        self.setLayout(thresholdLayout)

    def value(self):
        return self.slider.value() / 100


class LabeledSpinBox(QWidget):
    def __init__(self, text, delegate, initialValue, double=False):
        super().__init__()
        if double:
            self._sizeBox = QDoubleSpinBox()
            self._sizeBox.setDecimals(5)
        else:
            self._sizeBox = QSpinBox()
        self._sizeBox.setMinimum(0)
        self._sizeBox.setMaximum(9999)
        self._sizeBox.setValue(initialValue)
        self._sizeBox.valueChanged.connect(delegate)
        sizeLayout = QHBoxLayout()
        sizeLayout.addWidget(QLabel(text))
        sizeLayout.addWidget(self._sizeBox)
        self.setLayout(sizeLayout)

    def value(self):
        return self._sizeBox.value()


class Checkbox(QCheckBox):
    def __init__(self, text, delegate, initialValue):
        super().__init__()
        self.setText(text)
        self.setChecked(initialValue)
        self.stateChanged.connect(delegate)
