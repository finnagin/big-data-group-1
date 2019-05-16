import os
import random
import time

random.seed(time.time())

path = os.path.join(os.getcwd(),'g1_train')
test_path = os.path.join(os.getcwd(),'g1_test')

files = []
for r, d, f in os.walk(path):
    for file in f:
        if '.jpg' in file:
            files.append(os.path.join(r, file))

if len(files) == 17500:
    name_list = random.sample(files, 3500)
    for name in name_list:
        new_path = name.replace(path, test_path)
        os.rename(name, new_path)

