import argparse
from abc import ABC, abstractmethod
import datetime


class Program(ABC):
    def __init__(self):
        self._jobName = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        self._lastPrint = ""

    def JobName(self):
        return self.Name() + "_" + self._jobName
    
    @abstractmethod
    def Name(self):
        pass

    @abstractmethod
    def Description(self):
        pass

    def SetupParser(self, parser: argparse.ArgumentParser):
        pass

    def RunProgram(self, parserArgs: argparse.Namespace):
        pass

    def printRep(self, text=None):
        if text is None:
            print("")
            self._lastPrint = ""
        else:
            print("\b" * len(self._lastPrint) + text, end='\r')
            self._lastPrint = text
