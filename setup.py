from cx_Freeze import setup, Executable

# Dependencies are automatically detected, but it might need fine tuning.
# "packages": ["os"] is used as example only
build_exe_options = {"packages": ["PIL", "numpy", "skimage", "sklearn", "tflite_runtime"],
                     "excludes": ["tensorflow", "pandas", "matplotlib", "PySide6"],
                     "include_files": ["model/model.tflite",
                                       r"C:\Users\jonoj\miniconda3\envs\OrganoID\Library\bin\mkl_intel_thread.1.dll"]}

setup(
    name="OrganoID",
    version="1.0",
    description="Organoid image analysis software.",
    options={"build_exe": build_exe_options},
    executables=[Executable("OrganoID.py")]
)
