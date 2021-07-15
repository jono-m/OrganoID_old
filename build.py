from cx_Freeze import setup, Executable
import scipy
import skimage
from os import path

files = {"include_files": ['assets/',
                           path.dirname(skimage.__file__),
                           path.dirname(scipy.__file__)],
         "excludes": ["tensorflow"]}
setup(name="OrganoID",
      version="0.1",
      author="Jonathan Matthews",
      description="AI-driven organoid segmentation and tracking",
      options={'build_exe': files},
      executables=[Executable("main.py", base="Win32GUI", icon="assets/icon.ico", target_name="organoID.exe")]
      )
