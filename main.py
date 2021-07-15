from PySide6 import QtWidgets
from frontend.MainWindow import MainWindow
import sys
import os
from pathlib import Path

if __name__ == '__main__':
    os.chdir(Path(__file__).parent)
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()

    app.exec()
