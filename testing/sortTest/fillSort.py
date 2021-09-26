from pathlib import Path

directory = Path(".")

for xy in range(10):
    for num in range(4):
        fname = "testFile%03d_XY%03d_Zfield.txt" % (xy, num)
        (directory / fname).touch()