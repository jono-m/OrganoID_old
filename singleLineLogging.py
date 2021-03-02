class LoggerStatic:
    lastLogSize = 0


def ClearLog():
    LoggerStatic.lastLogSize = 0
    print("")


def DoLog(text: str):
    textLength = len(text)
    extraSpaces = " " * (LoggerStatic.lastLogSize - textLength)

    print("\t" + text + extraSpaces, end="\r", flush=True)

    LoggerStatic.lastLogSize = textLength
