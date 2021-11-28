REQUIREMENTS:

OrganoID was run with the following software configuration:
- Windows 10 64-bit
- Python 3.8

---

INSTALLATION:

To set up OrganoID dependencies, you could set up a venv environment first via 

>> pip install virtualenv
>> virtualenv OrganoIDVenv
>> source OrganoIDVenv/bin/activate

Then, you could run one of the following commands in your Python environment, depending on your use case. 

OrganoID standard:
>> pip install -r [\path\to\OrganoID\]requirements.txt

OrganoID with support for reproducing figures and statistics from publication:
>> pip install -r [\path\to\OrganoID\]requirementsFiguresAndData.txt

OrganoID with support for model training and data augmentation:
>> pip install -r [\path\to\OrganoID\]requirementsTrainingSuite.txt

---

INSTRUCTIONS:

OrganoID is run from the command line. To see instructions for use, run:

>> python OrganoID.py -h
