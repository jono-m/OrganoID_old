class Printer:
    _lastPrint = ""

    @staticmethod
    def printRep(text=None):
        if text is None:
            print("")
            Printer._lastPrint = ""
        else:
            print("\b" * len(Printer._lastPrint) + text, end='\r')
            Printer._lastPrint = text