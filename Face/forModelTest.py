# coding='utf-8'
import numpy as np
import openface
import os
import cv2
import pandas as pd

# dlib的人脸位置提取模型的路径
dlibFacePredictor = '/opt/openface/models/dlib/shape_predictor_68_face_landmarks.dat'  

networkModel = '/opt/openface/models/openface/nn4.small2.v1.t7'    
align = openface.AlignDlib(dlibFacePredictor)
net = openface.TorchNeuralNet(networkModel, 96)  # 神经网络输入图片尺寸96*96
cwd = os.getcwd()

# 这是提取训练集数据和标签，也就是基础集的特征点，然后用pandas处理保存到一个csv文件中，
#以备用查看，需要为每一个人创建一个文件夹，文件夹里面放上同一个人的图片，我的图片样本首先都按照
#照规则命名好了，因此可以直接将图片名称的字符串，split后提取出标签，在此说明。

def baseImageRep(imagesFileName):
    imagesFileName = imagesFileName
    personFileNameDir = os.path.join(cwd, imagesFileName)
    personFileName = os.listdir(personFileNameDir)

    labels = []
    dataSet = []
    for person in personFileName:
        personDir = os.path.join(personFileNameDir,person)
        imagesList = os.listdir(personDir)
        for image in imagesList:
            try:
                imagePath = os.path.join(personDir, image) # 目标图片的绝对路径

                img = cv2.imread(imagePath)
                if img is None:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                if img is None:
                    print 'img is None:%s' % image

                bb = align.getLargestFaceBoundingBox(img)
                alignedFace = align.align(96,img, bb,
                                          landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE)
                rep = net.forward(alignedFace)     # 提取目标文件的128位特征值

                dataSet.append(rep)
                label = image.split('_')[0]
                labels.append(label)

            except Exception as e:
                print image
        print '提取人物的特征点:', person

    assert len(dataSet) == len(labels)   # 样本数和标签数要相等
    dataSetDf = pd.DataFrame(dataSet,columns=map(lambda x: '特征值_%d'% x, range(1,129)))
    labelsDf = pd.DataFrame(labels,columns=['lables'])
    dataAllDf = pd.concat([labelsDf,dataSetDf], axis=1)
    print '正在生成基础集特征点文件  repBaseData.csv'
    dataAllDf.to_csv('repBaseData.csv')
    print  '文件保存完成   repBaseData.csv'
    dataSet = np.array(dataSet)   # 转化为numpy数组???需要吗？？？当然需要，这里坑了一下，kNN遍历计算距离时需要用到numpy array
    # labels = np.array(labels)   # labels没有进行计算，可以不转化为numpy array，直接return 一个list

    return dataSet, labels

# 这是提取测试集特征值
def testImagesRep(testImages):
    testImageFilePath = os.path.join(cwd, testImages)
    imagesList = os.listdir(testImageFilePath)
    dataTest = []
    labelsTest = []

    for image in  imagesList:
        imagePath = os.path.join(testImageFilePath, image)  # 目标图片的绝对路径
        try:
            img = cv2.imread(imagePath)
            if img is None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            if img is None:
                print 'img is None:%s' % image

            bb = align.getLargestFaceBoundingBox(img)
            alignedFace = align.align(96, img, bb,
                                      landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE)
            rep = net.forward(alignedFace)  # 提取目标文件的128位特征值

            dataTest.append(rep)
            labelTest = image.split('_')[0]
            labelsTest.append(labelTest)

        except Exception as e:
            print image

        finally:
            print '提取测试集中人物特征点', imagePath

    assert len(dataTest)==len(labelsTest)   # 断言样本数和labels数量相等

    datasSetTestDf = pd.DataFrame(dataTest, columns=map(lambda x: '特征值_%d' % x, range(1, 129)))
    labelsTestDf = pd.DataFrame(labelsTest, columns=['lables'])
    dataAllTestDf = pd.concat([labelsTestDf, datasSetTestDf], axis=1)
    print '正在生成测试样本集的特征值   repTestData.csv'
    dataAllTestDf.to_csv('repTestData.csv')
    print '文件完成保存    repTestData.csv'

    dataTest=np.array(dataTest)
    return dataTest, labelsTest

#k近邻分类器
def kNNClassify(inX, dataSet, labels, k=3):
    '''
    :param inX:  测试的样本128位特征值
    :param dataSet: 带标签基础集数据，128列的numpy数组
    :param labels: 基础集的标签，numpy array，List均可,只取数，没计算，不影响
    :param k: 就是著名的K了，自己根据每个人的样本数量选择吧
    :return: 预测标签label
    '''
    disTance = map(sum, np.power(dataSet-inX, 2)) # 计算欧几里得距离
    data = np.vstack((disTance, labels))  # 将数据和标签整合
    dataT = data.T
    dataT = dataT.tolist()

    for i in range(len(dataT)):
        dataT[i][0] = float(dataT[i][0])
    dataT.sort()

    count = dict()
    for i in range(k):
        if dataT[i][1] not in count:
            count[dataT[i][1]] = 1
        else:
            count[dataT[i][1]] +=1
  
    # 根据字典的value值，对字典进行排序，可以有这3种方法：
    # res = sorted(count.items(),key=lambda x:x[1],reverse=True)  # 方法1
    # res=sorted(count.items(), key=operator.itemgetter(1),reverse=True)  # 方法2
  
    res =zip(count.values(), count.keys())
    res = sorted(res, reverse=True)
    label = res[0][1]  # 取距离最近的一个tuple，第二个元素就是这个样本的标签
    return label
