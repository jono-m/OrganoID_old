from cx_Freeze import setup, Executable
import skimage
import scipy
import sklearn
from os import path


files = {"include_files": ['assets/',
                           path.dirname(skimage.__file__),
                           path.dirname(scipy.__file__),
                           path.dirname(sklearn.__file__)]}
setup(name="OrganoID",
      version="0.1",
      author="Jonathan Matthews",
      description="AI-driven organoid segmentation and tracking",
      options={'build_exe': files},
      executables=[Executable("main.py", base="Win32GUI", icon="assets/icon.ico", target_name="organoID.exe")]
      )
