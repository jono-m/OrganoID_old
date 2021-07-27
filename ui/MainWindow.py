from PySide6.QtWidgets import QMainWindow
from frontend.PipelineWidget import PipelineWidget
from PySide6.QtGui import QIcon


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        pipelineWidget = PipelineWidget()
        self.setCentralWidget(pipelineWidget)

        self.setWindowTitle("OrganoID")
        self.setWindowIcon(QIcon("assets/icon.png"))
        self.show()
        self.move(self.screen().geometry().center() - self.rect().center())
