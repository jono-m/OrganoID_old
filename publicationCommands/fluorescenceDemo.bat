:: Run this script from the OrganoID main directory.

python OrganoID.py detect model\model.tflite E:\FluoroFiles\*C1* -O E:\FluoroFiles\detected --heat
python OrganoID.py label E:\FluoroFiles\detected\*detected* -O E:\FluoroFiles\labeled --rgb -T 0 --edge
python OrganoID.py track E:\FluoroFiles\labeled\*labeled* E:\FluoroFiles\*C1* E:\FluoroFiles\tracked -B 1.5 --batch