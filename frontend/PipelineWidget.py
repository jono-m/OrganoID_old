import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from PIL import Image, ImageQt, ImageOps
from typing import Optional
from PySide6.QtWidgets import QFrame, QLabel, QPushButton, QFileDialog, QGridLayout, QVBoxLayout, QHBoxLayout, \
    QSizePolicy, QSlider, QCheckBox, QSpinBox, QWidget
from PySide6.QtGui import QPixmap, Qt, QPainter, QFont, QFontMetrics
from PySide6.QtCore import QPoint, QSize
import skimage.measure
import skimage.color
import numpy as np
from pathlib import Path
from backend.segmentation import OpenModel, SegmentImage, PostSegment
from backend.watershed import Watershed, PostProcess


class PipelineWidget(QFrame):
    def __init__(self):
        super().__init__()
        modelPath = Path(r"assets\trainedModel")
        self._model = OpenModel(modelPath)

        self._fileLabel = QLabel("Browse for file...")
        browseButton = QPushButton("Browse")
        browseButton.clicked.connect(self.BrowseForFile)

        layoutA = QVBoxLayout()
        layoutA.setAlignment(Qt.AlignTop)
        layoutB = QHBoxLayout()
        layoutB.setAlignment(Qt.AlignLeft)
        layoutC = QGridLayout()

        layoutB.addWidget(self._fileLabel)
        layoutB.addWidget(browseButton)
        layoutA.addLayout(layoutB)

        self._originalImageWidget = ImageWidget("Original")
        self._segmentedImageWidget = ImageWidget("Segmented")
        self._distanceTransformImageWidget = ImageWidget("Distance Transform")
        self._minimaImageWidget = ImageWidget("Local Maxima")
        self._watershedImageWidget = ImageWidget("Watershed Separation")
        self._finalImageWidget = ImageWidget("Post Processing")

        self._thresholdSlider = LabeledSlider("Threshold", self.UpdateSegmentationPost, 50)

        self._fgWatershedSlider = LabeledSlider("Opacity", self.UpdateWidgets, 100)
        self._fgFinalSlider = LabeledSlider("Opacity", self.UpdateWidgets, 20)

        self._holesCheckbox = QCheckBox("Fill Holes")
        self._holesCheckbox.stateChanged.connect(self.UpdateSegmentationPost)

        self._separationSpinBox = LabeledSpinBox("Minimum Separation", self.UpdateWatershedRaw, 20)

        layoutC.addWidget(self._originalImageWidget, 0, 0)

        layoutC.addWidget(self._segmentedImageWidget, 0, 1)
        layoutC.addWidget(self._thresholdSlider, 1, 1)
        layoutC.addWidget(self._holesCheckbox, 2, 1)

        layoutC.addWidget(self._distanceTransformImageWidget, 0, 2)

        layoutC.addWidget(self._minimaImageWidget, 0, 3)
        layoutC.addWidget(self._separationSpinBox, 1, 3)

        layoutC.addWidget(self._watershedImageWidget, 0, 4)
        layoutC.addWidget(self._fgWatershedSlider, 1, 4)

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

        self.UpdateOriginalRaw(Image.open(r"C:\Users\jonoj\Documents\ML\ch5f12a.jpg"))

    def BrowseForFile(self):
        filename, _ = QFileDialog.getOpenFileName(self, "Browse for Image")
        if filename:
            self._fileLabel.setText(filename)
            self.UpdateOriginalRaw(Image.open(filename))

    def UpdateOriginalRaw(self, original: Image):
        self._originalSize = original.size
        if original.mode == 'I' or original.mode == 'I;16':
            original = original.point(lambda x: x * (1 / 255))
        self._originalImageRaw = np.array(original)
        self.UpdateOriginalPost()

    def UpdateOriginalPost(self):
        self._originalImagePost = self._originalImageRaw
        self.UpdateSegmentationRaw()

    def UpdateSegmentationRaw(self):
        inputShape = self._model.layers[0].input_shape[0]
        preImage = Image.fromarray(self._originalImagePost).resize(inputShape[1:3]).convert(mode="L")
        imagePrepared = np.reshape(np.array(preImage), [1] + list(inputShape[1:]))
        self._segmentedImageRaw = SegmentImage(imagePrepared, self._model)
        self.UpdateSegmentationPost()

    def UpdateSegmentationPost(self):
        self._segmentedImagePost = PostSegment(self._segmentedImageRaw, self._thresholdSlider.value(),
                                               self._holesCheckbox.checkState())

        self.UpdateWatershedRaw()

    def UpdateWatershedRaw(self):
        self._dtImage, self._minimaImage, self._watershedImageRaw = Watershed(self._segmentedImagePost,
                                                                              self._separationSpinBox.value())

        self.UpdateWatershedPost()

    def UpdateWatershedPost(self):
        self._watershedImagePost = self._watershedImageRaw
        self.UpdateFinalRaw()

    def UpdateFinalRaw(self):
        self._finalImageRaw = PostProcess(self._watershedImageRaw, self.smallObjectsSpinBox.value(),
                                          self.cullSlider.value())
        self.UpdateFinalPost()

    def UpdateFinalPost(self):
        self._finalImagePost = self._finalImageRaw
        self.UpdateWidgets()

    def UpdateWidgets(self):
        self._originalImageWidget.SetImage(self._originalImagePost, self._originalImageRaw)
        self._segmentedImageWidget.SetImage(self._segmentedImagePost, self._originalImageRaw)
        self._watershedImageWidget.SetImage(self._watershedImagePost, self._originalImageRaw,
                                            imageAlpha=self._fgWatershedSlider.value(), label=True)
        self._finalImageWidget.SetImage(self._finalImagePost, self._originalImageRaw,
                                        imageAlpha=self._fgFinalSlider.value(), label=True)

        self._distanceTransformImageWidget.SetImage(self._dtImage, self._originalImageRaw, autoContrast=True)
        self._minimaImageWidget.SetImage(self._minimaImage, self._originalImageRaw, label=True)


class ImageWidget(QWidget):
    def __init__(self, title):
        super().__init__()

        layout = QVBoxLayout()
        self._imageLabel = QLabel()
        layout.addWidget(self._imageLabel)
        title = QLabel("<b>" + title + "</b>")
        title.setStyleSheet("""background-color: rgb(200, 200, 200)""")
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)
        self.setLayout(layout)

        self._labels = []

        self.SetImage(None)
        self._rawImage: Optional[QPixmap] = None
        self._imageLabel.setMinimumSize(256, 256)
        self._imageLabel.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

    def SetImage(self, image: np.ndarray = None, originalImage: np.ndarray = None, imageAlpha=1, label=False,
                 autoContrast=False):
        if image is None:
            preparedImage = Image.fromarray(np.zeros((512, 512))).convert(mode="L")
        else:
            height, width = originalImage.shape[:2]
            preparedImage = np.array(Image.fromarray(image).resize((width, height), resample=Image.NEAREST))
            if autoContrast:
                preparedImage = np.array(ImageOps.autocontrast(Image.fromarray(preparedImage).convert(mode="L")))
            self._labels = []
            if label:
                originalImage = np.array(Image.fromarray(originalImage).convert(mode="RGB"))
                preparedImage = skimage.color.label2rgb(preparedImage, originalImage, bg_label=0, alpha=imageAlpha,
                                                        image_alpha=1)
                preparedImage = Image.fromarray((preparedImage * 255).astype(np.uint8))
                props = skimage.measure.regionprops(image)
                for prop in props:
                    y, x = prop.centroid
                    oldHeight, oldWidth = image.shape
                    x, y = self.Transform((x, y), QSize(width, height), QSize(oldWidth, oldHeight))
                    self._labels.append(((x, y), str(prop.label)))
            else:
                preparedImage = Image.fromarray(preparedImage).convert(mode="L")

        qt = ImageQt.ImageQt(preparedImage)
        self._rawImage = QPixmap.fromImage(qt)

        self.doResize()

    def resizeEvent(self, event) -> None:
        self.doResize()

    def doResize(self):
        pixmap = self._rawImage.scaled(self.size(), Qt.KeepAspectRatio)
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

        self._imageLabel.setPixmap(pixmap)

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
    def __init__(self, text, delegate, initialValue):
        super().__init__()

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
