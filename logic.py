import os
path = 'D:/data/AD/data'
dirs = os.listdir(path)
for i in dirs:
    task = os.path.join(path, i)
    modals = os.listdir(task)
    for j in modals:
        k = os.path.join(task, j)
        os.mkdir(os.path.join(k, 'leaveoneout'))