import argparse
import re
from pathlib import Path

parser = argparse.ArgumentParser(
    description="Sorts directories by XY field of view. "
                "Will move any file in directory with XY[digits] in its name to a subfolder XY[digits]")

parser.add_argument("directory", help="Path to directory to sort", type=Path)
args = parser.parse_args()

directory: Path = args.directory
for file in directory.iterdir():
    matchData = re.match(r".*XY(\d+).*", file.stem)
    if matchData is None:
        continue
    dirName = "XY" + matchData[1]
    dirToMake = directory / dirName
    dirToMake.mkdir(parents=True, exist_ok=True)
    newPath = dirToMake / file.name
    file.rename(newPath)
