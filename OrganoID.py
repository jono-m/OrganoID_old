import argparse
from commandline.Train import Train
from commandline.Segment import Segment
from commandline.Augment import Augment
from commandline.Split import Split
from commandline.Performance import Performance
from commandline.PostProcess import PostProcess
from commandline.Track import Track

programs = [Train, Segment, Augment, Split, Performance, PostProcess, Track]

parser = argparse.ArgumentParser(
    description="Neural network segmentation and tracking of organoid microscopy images.")
subparsers = parser.add_subparsers(dest="subparser_name")

programs = [program() for program in programs]

for program in programs:
    program.SetupParser(subparsers.add_parser(program.Name(), help=program.Description()))

args = parser.parse_args()

for program in programs:
    if program.Name() == args.subparser_name:
        program.RunProgram(args)
        break
