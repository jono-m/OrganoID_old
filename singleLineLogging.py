class LoggerStatic:
    lastLogSize = 0


def ClearLog():
    LoggerStatic.lastLogSize = 0
    print("")


def DoLog(text: str):
    print(text)
