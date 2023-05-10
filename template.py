import os 

ROOT_DIR="src"
PROJECT_NAME="SchoolID_Prediction"
PROJECT_PATH=os.path.join(ROOT_DIR,PROJECT_NAME)

all_files_dir_list=[
    ROOT_DIR,
    PROJECT_PATH,
    os.path.join(ROOT_DIR,"__init__.py"),
    os.path.join(PROJECT_PATH,"__init__.py"),
    os.path.join(PROJECT_PATH,"code_dir"),
    os.path.join(PROJECT_PATH,"code_dir","__init__.py"),
    os.path.join(PROJECT_PATH,"all_models"),
    os.path.join(PROJECT_PATH,"all_models","__init__.py"),
    os.path.join(PROJECT_PATH,"all_csv"),
    os.path.join(PROJECT_PATH,"all_csv","__init__.py"),
    "init.sh",
    "setup.py",
    "ruuning_logs.log",
    "requirements.txt"
]


for item in all_files_dir_list:
    if '.' in os.path.splitext(item)[-1] and not os.path.exists(item):
        with open(item,'w') as f:
            pass
    else:
        if not os.path.exists(item):
            os.makedirs(item)