from os import listdir, walk, stat, makedirs
from os.path import isdir, isfile, join, exists

class Result():
    def setBasics(self, name, fp, fn, tp, tn, time):
        tmp = name.split('_')
        self.fp = fp
        self.fn = fn
        self.tp = tp
        self.tn = tn
        self.time = time
        self.pd = tp / (fn+tp+0.0)
        self.pf = fp / (tn+fp+0.0)
        self.f = 2*(self.pd*self.pf) / (self.pd+self.pf) if self.pd + self.pf > 0 else 0.0
        self.name = name
        self.preprocessing = tmp[7]
        self.bellweatherOrRoundRobin = tmp[3]
        self.featureSelection = tmp[1]
        self.undersampling = tmp[2]
        self.method = tmp[4]
        #self.method = tmp[4]+ "-" + tmp[5]
        self.metricType = tmp[0]
        self.localGlobal = tmp[6]

    def setStats(self, name, pd, pf, f, time):
        self.f = f
        self.pd = pd
        self.pf = pf
        self.time = time
        self.name = name
        self.preprocessing = tmp[7]
        self.bellweatherOrRoundRobin = tmp[3]
        self.featureSelection = tmp[1]
        self.undersampling = tmp[2]
        self.method = tmp[4]
        #self.method = tmp[4]+ "-" + tmp[5]
        self.metricType = tmp[0]
        self.localGlobal = tmp[6]


def dummyCheck(dir):
    num,num2, tot = 0,0,0
    num3 = []
    second = False
    # read in all data from all subfolders
    for d in listdir(dir):
        tot += 1
        if isdir(dir+d):
            tt = dummyCheck(dir+d+'/')
            num += tt[0]
            num2 += tt[1]
            tot += tt[2]
            num3.extend(tt[3])
        elif '.txt' in d:
            with open(dir+d, 'r') as cin:
                tmp = [c for c in cin]
            if len(tmp) > 0:
                num += 1
                second = True
    if second:
        num2 += 1
        num3.append(dir)

    return num, num2, tot, num3

def makeResults(file, raw=True):
    # read all lines from file
    tmpResults = []
    with open(file, 'r') as cin:
        tmpData = [c for c in cin]
    for td in tmpData:
        pieces = td.strip("\n").split(' ')
        numbers = pieces[-1].strip().split('-')
        if raw:
            # this assumes order fp, fn, tp, tn, time
            res = Result()
            res.setBasics(pieces[0], int(numbers[2]), int(numbers[1]), int(numbers[0]), int(numbers[3]), float(numbers[-1]))
            tmpResults.append(res)
        else:
            # this assumes order pd, pf, f, time
            res = Result()
            res.setStats(pieces[0], float(numbers[0]), float(numbers[1]), float(numbers[2]), float(numbers[-1]))
            tmpResults.append(res)

    return tmpResults

def readInputFiles(topDir, fileExtension, raw=True, lvlLimit=10):
    # raw = files contain tp, fn, fp, tn
    # not raw = files contain f, pf, pd
    dataset = []

    # get all files under the topDir which are not multiclass
    results = [join(root, f) for root, dirs, files in walk(topDir) for f in files
    if fileExtension in f if stat(join(root,f)).st_size != 0 if 'multiclass' not in f]

    """
    # version that is only multiclass
    results = [join(root, f) for root, dirs, files in walk(topDir) for f in files
    if fileExtension in f if stat(join(root,f)).st_size != 0 if 'multiclass' in f]
    """

    for r in results:
        dataset.extend(makeResults(r))

    return dataset

def makeSets(lst, divider):
    # undersampling 0.25, 0.5, 0.75, 1, 1.5, 2.0
    partitions = {str(d):[] for d in divider.values()[0]}
    criterion = divider.keys()[0]

    for l in lst:
        partitions[getattr(l, criterion)].append(l)

    for p in partitions.keys():
        print(p)
        print(len(partitions[p]))
    
    if not exists('./SemiProcessed/'+criterion):
        makedirs('./SemiProcessed/'+criterion)

    outF = open('./SemiProcessed/'+criterion+'/f.txt', 'w')
    outPD = open('./SemiProcessed/'+criterion+'/pd.txt', 'w')
    outPF = open('./SemiProcessed/'+criterion+'/pf.txt', 'w')
    outTime = open('./SemiProcessed/'+criterion+'/time.txt', 'w')

    for p in partitions.keys():
        outF.write(p+',')
        outF.write(','.join([str(item.f) for item in partitions[p]]))
        outF.write("\n")
        outPD.write(p+',')
        outPD.write(','.join([str(item.pd) for item in partitions[p]]))
        outPD.write("\n")
        outPF.write(p+',')
        outPF.write(','.join([str(item.pf) for item in partitions[p]]))
        outTime.write("\n")
        outTime.write(p+',')
        outTime.write(','.join([str(item.time) for item in partitions[p]]))
        outTime.write("\n")

    outF.close()
    outPD.close()
    outPF.close()
    outTime.close()


def analysis(resultsDir, metricsToConsider, fileType='.txt'):
    # build results (do this each time, or once?\

    # we'll do it once and see if there are performance issues, maybe make a
    # variant for large datasets which reads only what is needed each time
    print("reading data files")
    data = readInputFiles(resultsDir, fileType)
    """
    members = [attr for attr in dir(data[0]) if not callable(getattr(data[0], attr)) and not attr.startswith("__")]
    for m in members:
        print(m + ' : ' + str(getattr(data[0], m)))
    exit()
    """
    print("calculating statistics")
    for m in metricsToConsider.keys():
        makeSets(data, {m:metricsToConsider[m]})
            # divider (last) is {'undersampling': [0.25,0.5,0.75,1.0,1.5,2.0,5.0,10.0]}
    print("Commencing data analysis")
    print("or for now, just ending.")


"""
undersampling
preprocessing
method
metric type
bellweather vs roundrobin
local vs global (still need to figure out different definitions for "local" and run those)
feature selection

THE PLAN
----------------------
TESTING: local vs global
get all locals, compare them to the same thing except global
-- count number of times a better than b
--
"""
metrics = {'undersampling':[None, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 5.0, 10.0],
'preprocessing':['standardize', 'normalize'],
'method':['MLPClassifier', 'SVC', 'GaussianNB', 'KNeighborsClassifier', 'RandomForestClassifier',
'AdaBoostClassifier', 'LassoCV', 'ElasticNet', 'LogisticRegression', 'PCA', 'LinearDiscriminantAnalysis',
'OneClassSVM', 'DecisionTreeClassifier', 'OrthogonalMatchingPursuitCV', 'MultiTaskElasticNet',
'MultiTaskLasso', 'BayesianRidge', 'SGDClassifier', 'QuadraticDiscriminantAnalysis', 'GaussianProcessClassifier'],
'metricType':['PM', 'CK', 'MC'], 'bellweatherOrRoundRobin':['bellweather', 'roundrobin'],
'featureSelection':[None, 0.1,0.3,0.5,0.7,0.9,'kBest', 'autoSelect']}
top = '/Users/ahill6/Documents/Python/TransferLearning/Output2/'
analysis(top, metrics)

"""
a,b,c,d = dummyCheck('/Users/ahill6/Documents/Python/TransferLearning/Output/CK/')
print(a)
print(b)
print("Total: " + str(c))
#print(d)
"""
