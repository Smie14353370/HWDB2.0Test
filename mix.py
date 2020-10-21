import numpy as np
import cv2
import os
from glob import glob
import re
from tqdm import tqdm
 
def get_char_nums(segments):
    nums = []
    chars = []
    for seg in segments:
        label_head = seg.split('.')[0]
        label_name = label_head + '.txt'
        with open(os.path.join(label_root,label_name), 'r', encoding='gbk') as f:
            lines = f.readlines()
            nums.append(len(lines[0]))
            chars.append(lines[0])
    return nums, chars
 
def addZeros(s_):
    head, tail = s_.split('_')
    num = ''.join(re.findall(r'\d',tail))
    head_num = '0'*(4-len(num)) + num
    return head + '_' + head_num + '.jpg'
 
def strsort(alist):
    alist.sort(key=lambda i:addZeros(i))
    return alist
 
def pad(img, headpad, padding):
    assert padding>=0
    if padding>0:
        logi_matrix = np.where(img > 255*0.95, np.ones_like(img), np.zeros_like(img))
        ids = np.where(np.sum(logi_matrix, 0) == img.shape[0])
        if ids[0].tolist() != []:
            pad_array = np.tile(img[:,ids[0].tolist()[-1],:], (1, padding)).reshape((img.shape[0],-1,3))
        else:
            pad_array = np.tile(np.ones_like(img[:, 0, :]) * 255, (1, padding)).reshape((img.shape[0], -1, 3))
        if headpad:
            return np.hstack((pad_array, img))
        else:
            return np.hstack((img, pad_array))
    else:
        return img
 
def pad_peripheral(img, pad_size):
    assert isinstance(pad_size,tuple)
    w, h = pad_size
    result = cv2.copyMakeBorder(img, h, h, w, w, cv2.BORDER_CONSTANT, value=[255, 255, 255])
    return result
 
 
 
if __name__=='__main__':
    label_root = '_label'
    label_det = '_fullLabels'
    pages_root = '_images'
    pages_det = '_fullpages'
    os.makedirs(label_root, exist_ok=True)
    os.makedirs(pages_root, exist_ok=True)
    pages_for_set = os.listdir(pages_root)
    pages_set = set([pfs.split('_')[0] for pfs in pages_for_set])
    for ds in tqdm(pages_set):
        boxes = []
        pages = []
        seg_sorted = strsort([d for d in pages_for_set if ds in d])
        widths = [cv2.imread(os.path.join(pages_root, d)).shape[1] for d in seg_sorted]
        heights = [cv2.imread(os.path.join(pages_root, d)).shape[0] for d in seg_sorted]
        max_width = max(widths)
        seg_nums, chars = get_char_nums(seg_sorted)
        pad_size = (500, 1000)
        w, h = pad_size
        label_name = ds + '.txt'
        with open(os.path.join(label_det, label_name), 'w') as f:
            for i,pg in enumerate(seg_sorted):
                headpad = True if i==0 else True if seg_nums[i] - seg_nums[i-1]>5 else False
                pg_read = cv2.imread(os.path.join(pages_root, pg))
                padding = max_width - pg_read.shape[1]
                page_new = pad(pg_read, headpad, padding)
                pages.append(page_new)
                if headpad:
                    x1 = str(w + padding)
                    x2 = str(w + max_width)
                    y1 = str(h + sum(heights[:i+1]) - heights[i])
                    y2 = str(h + sum(heights[:i+1]))
                    box = np.array([int(x1),int(y1),int(x2),int(y1),int(x2),int(y2),int(x1),int(y2)])
                else:
                    x1 = str(w)
                    x2 = str(w + max_width - padding)
                    y1 = str(h + sum(heights[:i + 1]) - heights[i])
                    y2 = str(h + sum(heights[:i + 1]))
                    box = np.array([int(x1), int(y1), int(x2), int(y1), int(x2), int(y2), int(x1), int(y2)])
                boxes.append(box.reshape((4,2)))
                char = chars[i]
                f.writelines(x1 + ',' + y1 + ',' + x2 + ',' + y1 + ',' + x2 + ',' + y2 + ',' + x1 + ',' + y2 + ',' + char + '\n')
        pages_array = np.vstack(pages)
        pages_array = pad_peripheral(pages_array,pad_size)
        pages_name = ds + '.jpg'
        # cv2.polylines(pages_array, [box.astype('int32') for box in boxes], True, (0, 0, 255))
        cv2.imwrite(os.path.join(pages_det, pages_name),pages_array)
