from PySide6.QtWidgets import QApplication, QMainWindow, QLabel, QFrame, QHBoxLayout, QVBoxLayout, QSlider, QPushButton, \
    QMessageBox
from PySide6.QtGui import QPixmap, QImage, Qt, QMouseEvent
from PySide6.QtCore import QEvent, QTimer
from PIL import ImageFont, Image, ImageDraw
from backend.ImageManager import LoadImages, ComputeOutline, LabelTracks, SaveGIF
from skimage.measure import regionprops
import numpy as np
import sys
from pathlib import Path
from backend.Tracker import Tracker


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Labeler")

        self.rawLabeledImage = list(LoadImages(
            r"dataset\demo\labeled\20210127_organoidplate003_XY36_Z3_C2_detected_labeled.tiff"))[0]
        self.originalImage = list(LoadImages(
            r"dataset\demo\20210127_organoidplate003_XY36_Z3_C2.tif", (512, 512), "L"))[0]
        self.width, self.height = self.rawLabeledImage.frames[0].shape

        self.allRegionProps = [regionprops(frame) for frame in self.rawLabeledImage.frames]

        csvFile = open(r"dataset\demo\tracked\groundTruth\mapping.csv", "r")
        lines = csvFile.read().split("\n")[1:-1]
        self.labelToTrackIDMap = []
        for line in lines:
            frameNumber, rawLabel, mapping = line.split(", ")
            frameNumber = int(frameNumber)
            rawLabel = int(rawLabel)
            if frameNumber >= len(self.labelToTrackIDMap):
                self.labelToTrackIDMap.append({})
            self.labelToTrackIDMap[frameNumber][rawLabel] = mapping
        csvFile.close()

        self.trackedImages = []

        self.previousFrameWidget = QLabel()
        self.previousFrameWidget.setMouseTracking(True)
        self.previousFrameWidget.installEventFilter(self)
        self.previousFrameWidget.setFixedSize(self.width, self.height)
        self.currentFrameWidget = QLabel()
        self.currentFrameWidget.installEventFilter(self)
        self.currentFrameWidget.setFixedSize(self.width, self.height)
        self.previousFrameNumberWidget = QLabel()
        self.currentFrameNumberWidget = QLabel()

        self.slider = QSlider()
        self.slider.setMinimum(1)
        self.slider.setMaximum(len(self.rawLabeledImage.frames) - 1)
        self.slider.sliderMoved.connect(self.UpdateSlider)
        self.slider.setOrientation(Qt.Horizontal)

        self.saveButton = QPushButton("SAVE")
        self.saveButton.clicked.connect(self.Save)

        self.applyButton = QPushButton("APPLY")
        self.applyButton.setEnabled(False)
        self.applyButton.clicked.connect(self.Apply)

        mainLayout = QVBoxLayout()
        previousLayout = QVBoxLayout()
        previousLayout.addWidget(self.previousFrameWidget)
        previousLayout.addWidget(self.previousFrameNumberWidget)
        currentLayout = QVBoxLayout()
        currentLayout.addWidget(self.currentFrameWidget)
        currentLayout.addWidget(self.currentFrameNumberWidget)
        currentLayout.addWidget(self.applyButton)
        imagesLayout = QHBoxLayout()
        imagesLayout.addLayout(previousLayout)
        imagesLayout.addLayout(currentLayout)
        mainLayout.addLayout(imagesLayout)
        mainLayout.addWidget(self.slider)
        mainLayout.addWidget(self.saveButton)

        centralWidget = QFrame()
        centralWidget.setLayout(mainLayout)

        self.setCentralWidget(centralWidget)

        self.currentFrameNumber = 1
        self.selectedLabel = -1
        self.UpdateTrackedImages()
        self.UpdateSlider(1)

    def keyPressEvent(self, event) -> None:
        if self.selectedLabel > 0:
            maxTrackNumber = None
            for frameMap in self.labelToTrackIDMap:
                trackIDs = [int(frameMap[label]) for label in frameMap if frameMap[label].isdigit()]
                if not trackIDs:
                    continue
                maxTrackForFrame = max(trackIDs)
                if maxTrackNumber is None or maxTrackForFrame >= maxTrackNumber:
                    maxTrackNumber = maxTrackForFrame

            currentLabelText = self.labelToTrackIDMap[self.currentFrameNumber][self.selectedLabel]
            lastLabelText = currentLabelText

            if event.key() == Qt.Key.Key_Backspace:
                currentLabelText = currentLabelText[:-1]
            elif event.key() == Qt.Key.Key_Space:
                currentLabelText = str(maxTrackNumber + 1)
            elif event.text().isdigit():
                currentLabelText += event.text()

            if currentLabelText != lastLabelText:
                self.labelToTrackIDMap[self.currentFrameNumber][self.selectedLabel] = currentLabelText
                self.applyButton.setEnabled(True)
                self.UpdateCurrentView()

    def Apply(self):
        self.UpdateTrackedImages()
        self.UpdateTracksView()
        self.applyButton.setEnabled(False)

    def eventFilter(self, watched, event):
        if watched == self.currentFrameWidget and event.type() == QEvent.MouseButtonRelease and isinstance(event,
                                                                                                           QMouseEvent):
            lastLabel = self.selectedLabel
            self.selectedLabel = FindLabelForPoint(self.allRegionProps[self.currentFrameNumber], event.position())
            if self.selectedLabel != lastLabel:
                self.UpdateCurrentView()

        if watched == self.previousFrameWidget and event.type() == QEvent.MouseMove and isinstance(event,
                                                                                                   QMouseEvent):
            for frameNumber in reversed(range(self.currentFrameNumber)):
                regionProps = self.allRegionProps[frameNumber]
                labelMap = self.labelToTrackIDMap[frameNumber]
                label = FindLabelForPoint(regionProps, event.position())
                if label > 0:
                    print(labelMap[label])
                    break

        return super().eventFilter(watched, event)

    def UpdateSlider(self, value):
        if self.applyButton.isEnabled():
            ret = QMessageBox.question(self, "Confirm switching", "Unsaved changes. Do you want to switch images?")
            if ret == QMessageBox.No:
                self.slider.blockSignals(True)
                self.slider.setValue(self.currentFrameNumber)
                self.slider.blockSignals(False)
                return
        self.currentFrameNumber = value
        self.selectedLabel = -1
        self.UpdateTracksView()
        self.UpdateCurrentView()
        self.UpdateLabels()

    def BuildTracks(self):
        tracksByID = {}

        nextID = 0
        for (frameNumber, (regionProps, labelToTrackMap)) in enumerate(
                zip(self.allRegionProps, self.labelToTrackIDMap)):
            detectedTracks = []
            for rp in regionProps:
                trackID = labelToTrackMap[rp.label]
                if trackID == "":
                    continue
                trackID = int(trackID)
                if trackID in tracksByID:
                    if tracksByID[trackID] in detectedTracks:
                        # This one has already been detected!
                        continue
                else:
                    tracksByID[trackID] = Tracker.OrganoidTrack(frameNumber, nextID)
                    nextID += 1
                tracksByID[trackID].Detect(rp.centroid, rp.area, rp.coords, rp.image, rp.bbox, rp.label)
                detectedTracks.append(tracksByID[trackID])

            [tracksByID[trackID].NoDetection() for trackID in tracksByID if tracksByID[trackID] not in detectedTracks]

        for trackID in tracksByID:
            tracksByID[trackID].id = trackID

        tracksByID = [tracksByID[trackID] for trackID in tracksByID]
        return tracksByID

    def UpdateTrackedImages(self):
        tracksByID = self.BuildTracks()
        self.trackedImages = LabelTracks(tracksByID, (255, 255, 255, 255), 255, 50, (0, 205, 108), {},
                                         self.originalImage.frames)

    def UpdateLabels(self):
        self.previousFrameNumberWidget.setText("Previous: %d" % (self.currentFrameNumber - 1))
        self.currentFrameNumberWidget.setText("Current: %d" % self.currentFrameNumber)

    def Save(self):
        csvFile = open(r"dataset\demo\tracked\groundTruth\mapping.csv", "w+")
        csvFile.write("Frame, Original Label, Organoid ID\n")
        for frameNumber, mapping in enumerate(self.labelToTrackIDMap):
            for rawLabel in mapping:
                csvFile.write("%d, %d, %s\n" % (frameNumber, rawLabel, mapping[rawLabel]))
        csvFile.close()
        SaveGIF(self.trackedImages, Path(r"dataset\demo\tracked\groundTruth\groundTruth.gif"))

    def UpdateTracksView(self):
        image = self.trackedImages[self.currentFrameNumber - 1]
        [height, width, _] = image.shape
        bpl = width * 3
        qimage = QImage(image.data, width, height, bpl, QImage.Format.Format_RGB888)
        self.previousFrameWidget.setPixmap(QPixmap(qimage))

    def UpdateCurrentView(self):
        image = MakeLabelImage(self.rawLabeledImage.frames[self.currentFrameNumber],
                               self.originalImage.frames[self.currentFrameNumber],
                               self.allRegionProps[self.currentFrameNumber],
                               self.labelToTrackIDMap[self.currentFrameNumber], self.selectedLabel)
        [height, width, _] = image.shape
        bpl = width * 3
        qimage = QImage(image.data, width, height, bpl, QImage.Format.Format_RGB888)
        self.currentFrameWidget.setPixmap(QPixmap(qimage))


def FindLabelForPoint(regionProps, point):
    matches = [regionProp.label for regionProp in regionProps if [point.y(), point.x()] in regionProp.coords.tolist()]
    if matches:
        return matches[0]
    else:
        return -1


def MakeLabelImage(rawLabeledImage, originalImage, regionProps, labelMap, selectedLabel):
    font = ImageFont.truetype("arial.ttf", 26)

    pilImage = Image.new(mode="RGBA", size=rawLabeledImage.shape, color=(0, 0, 0, 0))
    drawer = ImageDraw.Draw(pilImage)

    # Draw each present track on the frame
    for region in regionProps:
        label = labelMap[region.label]

        fillCoords = list(zip(list(region.coords[:, 1]), list(region.coords[:, 0])))

        if region.label == selectedLabel:
            color = (255, 255, 255)
        else:
            if label == "":
                label = "?"
                color = (255, 0, 0)
            else:
                color = (0, 255, 0)

        drawer.point(fillCoords, color + (100,))
        borderCoords = ComputeOutline(region.image)
        globalCoords = borderCoords + region.bbox[:2]
        xs = list(globalCoords[:, 1])
        ys = list(globalCoords[:, 0])
        outlineCoords = list(zip(xs, ys))
        drawer.point(outlineCoords, color + (255,))

        y, x = list(region.centroid)
        drawer.text((x, y), str(label), anchor="ms", fill=(255, 255, 255, 255), font=font)

    baseImage = Image.fromarray(originalImage).convert(mode="RGBA")
    pilImage = Image.alpha_composite(baseImage, pilImage).convert(mode="RGB")
    return np.asarray(pilImage)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()

    app.exec()
