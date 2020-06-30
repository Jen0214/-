import numpy as np #数组相关的数据包
import os #操作文件系统的数据包

if __name__ == '__main__':
    print ("step 1: load data...") 
    train_x, train_y, test_x, test_y = DataSetLoad() #加载并生成训练集、训练集标签，测试集、测试集标签
 
    numTestSamples = test_x.shape[0] #测试集行数
    matchCount = 0 
    for i in range(numTestSamples): #遍历测试集中每一个数组
        predict = kNNClassify(test_x[i], train_x, train_y, 3)#进行KNN算法分类 
        if predict == test_y[i]:
            matchCount += 1 
    
    accuracy = float(matchCount) / numTestSamples #计算准确率
    print ("step 2: show the result...") 
    print ('The classify accuracy is: %.2f%%' % (accuracy * 100))


def img2vector(filename): #将二维数据转换为一维数组，filename为文件路径名
    rows = 32 #原数据是32行
    cols = 32 #原数据是32列
    imgVector = np.zeros((1, rows * cols)) #创建一个初始化是零的1行1024列的数组
    fileIn = open(filename) #打开文件
    for row in range(rows): #按行遍历这里是32，执行32次
        lineStr = fileIn.readline() #读取当前数据中的每一行
    for col in range(cols): # 遍历当前行中的每一列数据
        imgVector[0, row * 32 + col] = int(lineStr[col]) #将当前行的每一个数据一次放到imgVector数组中，即将二维数据变成一维数组 
    return imgVector #函数返回数组
 
def DataSetLoad(): #加载数据函数
    print ("---Getting training set...") 
    dataSetDir = './Mnist_Dataset/' #数据集的根目录，这种表示是相对路径
    trainingFileList = os.listdir(dataSetDir + 'trainingDigits') #将数据集目录中训练集的文件名放到列表中，列表中每一行元素
    numSamples = len(trainingFileList) #列表的长度，也是列表的行数，由于列表中每行存储的是一个文件名，所以也是文件的数量
 
    train_x = np.zeros((numSamples, 1024)) #创建一个列表，行数是列表的行数（文件的数量），列数是1024，它的每一行表征的是一个文件数据
    train_y = [] #标签0~9这九个数字
 
    for i in range(numSamples): #将训练集目录中每个文件的类别标签提取到train_y这个列表中
        filename = trainingFileList[i] 
        train_x[i, :] = img2vector(dataSetDir + 'trainingDigits/%s' % filename) #训练集列表中每一行存储一个训练集的文件 
        label = int(filename.split('_')[0]) #提取训练集文件名的第一个字符作为类别标签，由于文件名是"0_0"或者"0_1"这样的格式下划线前面表示类别
        train_y.append(label) 
 
    print ("---Getting testing set...") #测试集同训练集一样的步骤
    testingFileList = os.listdir(dataSetDir + 'testDigits') 
    numSamples = len(testingFileList)
 
    test_x = np.zeros((numSamples, 1024)) 
    test_y = [] 
 
    for i in range(numSamples): 
        filename = testingFileList[i] 
        test_x[i, :] = img2vector(dataSetDir + 'testDigits/%s' % filename) 
        label = int(filename.split('_')[0]) # return 1 
        test_y.append(label) 
 
    return train_x, train_y, test_x, test_y #返回训练集列表，训练集列表标签，测试集列表，测试集列表标签
    
def kNNClassify(newInput, dataSet, labels, k): #knn分类算法 newInput：待测试的测试集数字。dataSet：整个训练集列表。labels：训练集标签列表。k: 取k个相近的
    numSamples = dataSet.shape[0] #训练集列表的行数
 
    diff = np.tile(newInput, (numSamples, 1)) - dataSet #输入的数据列数不变，行数变成numSamples，即将原数组复制numSamples份
    squaredDiff = diff ** 2                             #计算欧式距离
    squaredDist = np.sum(squaredDiff, axis = 1)
    distance = squaredDist ** 0.5
    sortedDistIndices = np.argsort(distance)            #将计算后的距离由小到大排序，提取排序后的索引值列表放到sortedDistIndices
 
    classCount = {} #创建字典
    for i in range(k): 
        voteLabel = labels[sortedDistIndices[i]]#取索引列表前k个值遍历，即距离最小的前k个点，记录其类别
        classCount[voteLabel] = classCount.get(voteLabel, 0) + 1 #投票

    maxCount = 0  #找到投票数量最多的标签类别
    for key, value in classCount.items():
        if value > maxCount: 
            maxCount = value
            maxIndex = key 
    return maxIndex 
