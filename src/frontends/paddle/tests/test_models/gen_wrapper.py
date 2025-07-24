# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import glob
import os
import subprocess
import paddle
import sys

print(sys.argv)
if len(sys.argv) < 3:
    print("Gen folder and output folder must be specified as arguments")
    sys.exit(1)

gen_folder = sys.argv[1]
out_folder = sys.argv[2]
mark_file = os.path.join(out_folder, "generate_done.txt")

gen_files = glob.glob(os.path.join(gen_folder, '**/generate_*.py'), recursive=True)
failed_filelist = []

enable_pir = os.getenv('FLAGS_enable_pir_api')
if  enable_pir == "1":
    failed_filelist = [
    "generate_assign.py",
    "generate_place_test_model.py"
    ]

for gen_script in gen_files:
    print("Processing: {} ".format(gen_script))
    torun = True;
    for file in failed_filelist:
        if file in gen_script:
            torun = False
            break
    if torun == False:
        continue
    status = subprocess.run([sys.executable, gen_script, out_folder], env=os.environ)
    if status.returncode != 0:
        print("ERROR: PaddlePaddle model gen script FAILED: {}".format(gen_script))
        sys.exit(1)

# Create mark file indicating that script was executed
with open(mark_file, "w") as fp:
    pass
