# Program.py -- a generic class for a sub-program of OrganoID.

import argparse
from abc import ABC, abstractmethod


class Program(ABC):
    def __init__(self):
        self._lastPrint = ""

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