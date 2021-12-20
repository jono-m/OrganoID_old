:: Run this script from the OrganoID main directory.

python OrganoID.py detect model\model.tflite dataset\demo -O dataset\demo\detected --heat
python OrganoID.py label dataset\demo\detected\*detected* -O dataset\demo\labeled --rgb -T 0 --edge
python OrganoID.py track dataset\demo\labeled\*labeled* dataset\demo dataset\demo\tracked -B 1.5 --individual