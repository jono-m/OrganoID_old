REQUIREMENTS:

OrganoID was run with the following software configuration:
- Windows 10 64-bit
- Python 3.8


<b>The compiled executable can be downloaded <a href="https://drive.google.com/drive/folders/1xFUkUANFvqCjudQk7SYDj7uY7xVvmUJP?usp=sharing">here</a>.</b>

---

SOURCE CODE INSTALLATION:

To set up OrganoID source dependencies, you could set up a venv environment first via

>> pip install virtualenv
>> virtualenv OrganoIDVenv
>> cd OrganoIDVenv/Scripts/
>> activate.bat

Then, you could run one of the following commands in your Python environment, depending on your use case. 

OrganoID standard:
>> pip install -r [\path\to\OrganoID\]requirements.txt

OrganoID with support for reproducing figures and statistics from publication:
>> pip install -r [\path\to\OrganoID\]requirementsFiguresAndData.txt

OrganoID with support for model training and data augmentation:
>> pip install -r [\path\to\OrganoID\]requirementsTrainingSuite.txt

---

INSTRUCTIONS:

OrganoID is currently only available as a command line tool. To see instructions for use, run:

>> python OrganoID.py -h

A PySide6 graphical interface for OrganoID and an ImageJ plugin are in development with expected release mid-2022.
