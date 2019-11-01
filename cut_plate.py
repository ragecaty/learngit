from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import pandas as pd
import time
import shutil


def areaFilter(boxes):

    clip_area = []  # 限定box大小
    for i in boxes:
        if all([i[2] < 45, i[2] > 5, i[3] < 80, i[3] > 45]):
            clip_area.append(i)
    return clip_area


def sameFilter(clip_area):

    d = {} # 删除相同的box
    for i in clip_area:
        if i[0] not in d:
            d[i[0]] = [i]
        else:
            d[i[0]].append(i)
    sameArea = []  # 保留相同区域的最大范围
    for i in d:
        max1 = [d[i][0][1], 0, 0]
        for j in d[i]:
            if max1[0] > j[1]:
                max1[0] = j[1]
            if max1[1] < j[2]:
                max1[1] = j[2]
            if max1[2] < j[3]:
                max1[2] = j[3]
        sameArea.append([i, *max1])
    return sameArea  # sameArea is a list


def meanFilter(sameArea):

    df = pd.DataFrame(sameArea, columns=['x', 'y', 'dx', 'dy'])

    # 均值过滤 （解决长宽过大问题）
    minHeight = np.mean(df['dy']) - 11
    maxHeight = np.mean(df['dy']) + 11

    df = df[(df['dy'] < maxHeight) & (df['dy'] > minHeight) & (df['dx'] > 15)]

    # 遇到字符1比较扁，容易误删
    # minWeight = np.mean(df['dx'])-10
    # maxWeight = np.mean(df['dx'])+10
    # df3 = df2[(df2['dx']<maxWeight) & (df2['dx']>minWeight)]

    df = df.sort_values(by='x')
    df.index = [i for i in range(len(df))]
    boxes = np.array(df[['x', 'y', 'dx', 'dy']])

    return boxes


def forwardFilter(boxes):

    fWidth = []
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            fwidth = boxes[i][0] - boxes[j][0]  # 前向宽度 x1-x1
            farea = boxes[i][2] * boxes[i][3]  # 面积对比，删除小的
            barea = boxes[j][2] * boxes[j][3]
            if -5 < fwidth < 5:
                if farea < barea:
                    fWidth.append(i)
                if farea >= barea:
                    fWidth.append(j)
            break
    return np.delete(boxes, fWidth, 0)


def backwardFilter(boxes):

    bWidth = []
    flag = False
    for i in range(len(boxes)):
        if flag:  # 循环跳过判断
            flag = False
            continue
        for j in range(i + 1, len(boxes)):
            width = boxes[j][0] - boxes[i][0] - boxes[i][2]  # x2 - (x1+w1)
            if width < -1:
                bWidth.append(j)
                flag = True
            break
    return np.delete(boxes, bWidth, 0)


def clip_main(clip_dir):

    for pic in clip_dir:
        im = Image.open('ques/' + pic)
        im = im.convert('RGB')
        arr = np.array(im)
        arr = cv2.resize(arr, (300, 150))

        gray = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
        mser = cv2.MSER_create(_min_area=100, _max_variation=0.2)
        regions, boxes = mser.detectRegions(gray)

        # start cutting character
        clip_area = areaFilter(boxes)
        sameArea = sameFilter(clip_area)
        boxes1 = meanFilter(sameArea)  # 均值过滤
        boxes2 = forwardFilter(boxes1)  # 前向过滤
        boxes3 = forwardFilter(boxes2)  # 第二次前向
        boxes4 = backwardFilter(boxes3)  # 后向

        os.makedirs('clip_plate/' + pic)  # 切割字符保存路径
        iter1 = 0
        for i in boxes4:
            try:
                x, y, w, h = i
                pic1 = arr[max(0, y - 2):min(y + h + 2, 140), max(0, x - 2):min(300, x + w + 2), :]
                cv2.imwrite(f'clip_plate/{pic}/{iter1}.jpg', pic1)
                iter1 += 1
            except:
                with open('exception.txt', 'a+') as f:
                    f.write(pic + '\n')

if __name__ == '__main__':

    clip_dir = os.listdir('ques/')  # 所有车牌路径
    clip_main(clip_dir)
