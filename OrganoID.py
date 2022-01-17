# OrganoID.py -- Entry-point for the OrganoID suite.

import argparse
from commandline.Augment import Augment
from commandline.Detect import Detect
from commandline.Analyze import Analyze
from commandline.Label import Label
from commandline.Split import Split
from commandline.Track import Track
from commandline.Train import Train

# List of sub-programs.
programs = [Augment, Detect, Label, Split, Track, Train, Analyze]

# Parse sub-program selection
parser = argparse.ArgumentParser(
    description="OrganoID: deep learning for organoid image analysis.")
subparsers = parser.add_subparsers(dest="subparser_name")

# Instantiate sub-programs
programs = [program() for program in programs]

# Load sub-program arguments
for program in programs:
    program.SetupParser(subparsers.add_parser(program.Name(), help=program.Description()))

# Parse all command-line arguments.
args = parser.parse_args()

# Run the selected sub-program with the parsed arguments.
for program in programs:
    if program.Name() == args.subparser_name:
        program.RunProgram(args)
        break
