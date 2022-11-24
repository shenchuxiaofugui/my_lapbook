import os,shutil
import gzip
filepath = '/homes/syli/dataset/zj_data/jly/data'


def unzip_gz(file_name):
    # 获取文件的名称，去掉后缀名
    f_name = file_name.replace(".gz", "")
    # 开始解压生成一个类
    g_file = gzip.GzipFile(file_name)
    # 读取数据部分(字节流）写入一个文件
    open(f_name, "wb+").write(g_file.read())
    g_file.close()



def copy_one(ori_img, ori_roi,newpath):
    if not os.path.exists(newpath + '.nii'):
        shutil.copy(ori_img, newpath + '.nii.gz')
        unzip_gz(newpath + '.nii.gz')
    if not os.path.exists(newpath + '_roi.nii'):
        shutil.copy(ori_roi, newpath + '_roi.nii')
        os.remove(newpath + '.nii.gz')



def copy(ori_img, ori_roi, newpath):

    if 't1' in ori_img:
        if '_c' in ori_img or '_vibe' in ori_img:
            datapath = newpath + '/t1_c'
            copy_one(ori_img, ori_roi, datapath)
        else:
            datapath = newpath + '/t1'
            copy_one(ori_img, ori_roi, datapath)
    elif 'ADC' in ori_img:
        datapath = newpath + '/ADC'
        copy_one(ori_img, ori_roi, datapath)
    elif 't2' in ori_img:
        if ('_c' not in ori_img) and ('_vibe' not in ori_img):
            datapath = newpath + '/t2'
            copy_one(ori_img, ori_roi, datapath)


def new_data(filepath):
    dirs = os.listdir(filepath)
    for i in dirs:
        newpath = '/homes/syli/dataset/zj_data/jly/clear_up/' + i
        #os.mkdir(newpath)
        files = os.path.join(filepath, i)
        filelist = [os.path.join(files, j) for j in os.listdir(files) if 'nii' in j]
        filelist.sort()
        candidate_img = [i for i in filelist[1::2]]
        for ori_img in candidate_img:
            ori_roi = ori_img[:-4] + '-label.nii'
            copy(ori_img, ori_roi, newpath)

def old_data(filepath):
    dirs = os.listdir(filepath)
    for i in dirs:
        newpath = '/homes/syli/dataset/zj_data/jly/clear_up/' + i
        #os.mkdir(newpath)
        files = os.path.join(filepath, i)
        filelist = [os.path.join(files, j) for j in os.listdir(files) if ('BSpline' in j and 'b1000' in j)]
        ori_roi = [os.path.join(files, j) for j in os.listdir(files) if ('label.nii' in j and 't1' in j)]
        try:
            ori_img = filelist[0]
            roi_img = ori_roi[0]
            copy_one(ori_img, roi_img, newpath + '/b1000')
        except (Exception, BaseException) as e:
            print(files)
            print(e)


def remove():
    dirs = ['jihongyi', 'qianjinxiu']
    for i in dirs:
        path = '/homes/syli/dataset/zj_data/jly/clear_up/' + i
        files = os.listdir(path)
        files = [j for j in files if '.gz' in j]
        for j in files:
            try:
                os.remove(os.path.join(path, j))
            except:
                print(i)

old_data(filepath)
