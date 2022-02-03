:: Run this script from the OrganoID main directory.

build\exe.win-amd64-3.8\OrganoID.exe detect build\exe.win-amd64-3.8\model.tflite dataset\demo -O dataset\demo\detected --heat
build\exe.win-amd64-3.8\OrganoID.exe label dataset\demo\detected\*detected* -O dataset\demo\labeled --rgb -T 0 --edge
build\exe.win-amd64-3.8\OrganoID.exe track dataset\demo\labeled\*labeled* dataset\demo dataset\demo\tracked -B 1.5 --individual