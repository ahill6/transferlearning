from sklearn.svm import SVC, OneClassSVM
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LassoCV, ElasticNet, LogisticRegression,OrthogonalMatchingPursuitCV, MultiTaskElasticNet, MultiTaskLasso, BayesianRidge, SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold, cross_val_score, GridSearchCV, train_test_split
from sklearn.feature_selection import SelectKBest, RFECV, GenericUnivariateSelect, VarianceThreshold, mutual_info_classif
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.cluster import AgglomerativeClustering, DBSCAN
from sklearn.metrics import make_scorer, accuracy_score, auc, confusion_matrix,f1_score,precision_recall_fscore_support
from sklearn.preprocessing import StandardScaler, Normalizer
from sklearn.decomposition import PCA

from warnings import catch_warnings, simplefilter
from os import makedirs
from os.path import dirname
#import autosklearn
from sys import exit
from copy import deepcopy
from timeit import timeit
from time import time
from random import shuffle, sample
from math import ceil
#from contextlib import contextmanager
#from spatialtree import spatialtree
import numpy

# TODO - Go through this and figure out what needs to stay

def percentUndersampling(data, labels, rate, keepRatioMinority=1):
    """Does random undersampling of majority class and returns a subset of the original data with the size of the majority
    class instances = size minority instances * rate.
    NOTE - this assumes only two classes (0 and not 0).
    data    - training data
    labels  - class labels for data
    rate    - ratio of minority/majority class size desired
    """
    p = [data[d] for d in range(len(data)) if labels[d] != 0] # minority class instances
    n = [data[d] for d in range(len(data)) if labels[d] == 0] # majority class instances

    # take size(p)*keepRatio members of the minority class and n*rate (round up) members of majority class
    out = sample(p, ceil(len(p)*keepRatioMinority))
    out.extend(sample(n, min(ceil(len(out)*rate), len(n))))

# TODO - make this method
def processUndersampling(data, rate):
    print("?")

def timer(f):
    t1 = time()
    f()
    t2 = time()
    return t2 - t1

def prepareData(X, y, importantIndex, underSampling=None, featureSelection=None, ordering='roundrobin', preprocess=None, mini=None):
    # TODO - shuffle the data (X and y same if have both)
    tmpX, tmpY, tmpOthersX, tmpOthersY = X[importantIndex], y[importantIndex], [], []

    for x in range(len(X)):
        if x != importantIndex:
            if ordering == 'roundrobin':
                tmpOthersY.extend(y[x])
                tmpOthersX.extend(X[x])
            elif ordering == 'bellweather':
                tmpOthersY.append(y[x])
                tmpOthersX.append(X[x])

    if mini is not None:
        if not 0 <= mini <= 1:
            print("invalid mini value (must be [0,1]), disregarding")
            mini = None
        elif underSampling is not None:
            print("You can't do both mini and undersampling.  Just doing mini")
            underSampling = None

    if preprocess is not None:
        if preprocess == 'normalize':
            tmpX = Normalizer().fit_transform(tmpX)
            if ordering =='bellweather':
                for x in range(len(tmpOthersX)):
                    tmpOthersX[x] = Normalizer().fit_transform(tmpOthersX[x])
            else:
                tmpOthersX = Normalizer().fit_transform(tmpOthersX)
        elif preprocess == 'standardize':
            tmpX = StandardScaler().fit_transform(tmpX)
            if ordering == 'bellweather':
                for x in range(len(tmpOthersX)):
                    tmpOthersX[x] = StandardScaler().fit_transform(tmpOthersX[x])
            else:
                tmpOthersX = StandardScaler().fit_transform(tmpOthersX)

    if featureSelection is not None:
        if isinstance(featureSelection, float):
            # removes any features with low variance (if 90% of entries have a 0, likely that feature isn't important)
            try:
                tran = VarianceThreshold(threshold=(featureSelection*(1-featureSelection))).fit(tmpX) # not sure this is done right
                tmpX = tran.transform(tmpX)
            except ValueError as v:
                # need to add here a check that "No feature in X meets the variance threshold" is the error
                print("feature selection failure ", featureSelection, v)
                return -1,-1,-1,-1
            if ordering == 'bellweather':
                for x in range(len(tmpOthersX)):
                    tmpOthersX[x] = tran.transform(tmpOthersX[x])
            else:
                tmpOthersX = tran.transform(tmpOthersX)
        elif featureSelection == 'kBest':
            # select K Best (removes all but K 'best' features, as measured by score in XXX
            tran = SelectKBest(mutual_info_classif, 5).fit(tmpX, tmpY)
            tmpX = tran.transform(tmpX) # picked both 'best' metric and k by gut...
            if ordering == 'bellweather':
                for x in range(len(tmpOthersX)):
                    tmpOthersX[x] = tran.transform(tmpOthersX[x])
            else:
                tmpOthersX = tran.transform(tmpOthersX)
            #chi2, f_classif, mututal_info_classif (last is non-parametric, others not), k
        elif featureSelection == 'autoSelect':
            tran = GenericUnivariateSelect(mutual_info_classif).fit(tmpX, tmpY)
            tmpX = tran.transform(tmpX)
            if ordering == 'bellweather':
                for x in range(len(tmpOthersX)):
                    tmpOthersX[x] = tran.transform(tmpOthersX[x])
            else:
                tmpOthersX = tran.transform(tmpOthersX)

        # RFECV is a feature selection method which takes a model and uses that to reduce the features.
        # TODO - for bellweather, take feature selection trained on the model with the best performance over the training data

    if mini is not None:
        # randomly sample mini % of the data.
        indexes = shuffle(list(range(len(tmpX))), min(max(len(tmpX)*mini, 25), len(tmpX)))
        tmpX, tmpY = [i for i, x in enumerate(tmpX) if x in indexes], [i for i, x in enumerate(tmpY) if x in indexes] # need a version of this that guarantees at least one minority class sample
    elif underSampling is not None:
        if isinstance(underSampling, float):
            tmpX = percentUndersampling(tmpX, tmpY, underSampling)
        else:
            tmpX = processUndersampling(tmpX, tmpY, underSampling)

    return tmpX, tmpY, tmpOthersX, tmpOthersY


def readCsv(path, filenames, groundTruth=True, removeDuplicates=False, recordStatistics=False):
    first = True
    data = [[] for _ in filenames]
    classes = [[] for _ in filenames]
    stats, tmp, tmp1= [], [], []
    i = 0
    for f in filenames:
        with open(path+f, 'r') as temp_file:
            if first:
                labels = temp_file.readline()
            else:
                temp_file.readline()
            tmp = [[float(r) for r in line.rstrip("\n").split(',')] for line in temp_file]

        if recordStatistics:
            meds = numpy.median(tmp, axis=0)
            mins = numpy.amin(tmp, axis=0)
            maxs = numpy.amax(tmp, axis=0)
            stds = numpy.std(tmp, axis=0)
            cov = numpy.corrcoef(tmp, rowvar=0)
            evals, evects = numpy.linalg.eig(cov)
            ev = numpy.unique(evects)
            data2 = numpy.dot(tmp, evects)
            corr = numpy.corrcoef(data2, rowvar=0)
            stats.append([meds, mins, maxs, stds, cov, ev, corr])  # this is likely not a good way to do it because
            # some of these are vectors, others matrices

            # maybe add something quantifying the amount of duplication in the dataset?
            # TODO - yes, esp. for comparison when using mini

            # output stats to file here

        if removeDuplicates:
            for t in tmp:
                if t not in tmp1[i]:
                    tmp1.append(t)
        else:
            tmp1 = deepcopy(tmp)


        if groundTruth:
            tmpX, tmpy = [], []
            for t in tmp1:
                data[i].append(t[:-1])
                classes[i].append(t[-1])
        else:
            data[i] = deepcopy(tmp1)

        i += 1

    return data, classes


def transferLearning(files, filepathStarter, inpath='.', outpath='.', undersample=-1, isLocal=False, strategy='roundrobin', preprocessing='normalize',
                     classifiersettings=False, featureselection=None):
    runtime = {}
    settingsLookup = {}

    loc = "local" if isLocal else "global"
    cla = "all" if classifiersettings else "some"
    pre = preprocessing if preprocessing is not None else "none"
    # FORMAT OF : MC\\kBest\\0.25\\bellweather\\local\\
    #           dataset\featSel\underSamp\bellweather vs round robin\local vs global\
    absoluteOutPath = outpath + filepathStarter + "\\" + str(featureselection) + "\\" + str(
        undersample) + "\\" + str(strategy) + "\\" + loc + "\\" + pre + "\\" + cla + "\\"

    # FORMAT OF : MC_kBest_0.25_bellweather_knn_1_local_standardize
    #   dataset_featSel_underSamp_bellweather vs round robin_classifier_classifier settings_local vs global_preprocessing option
    methodDetails = filepathStarter + "_" + str(featureselection) + "_" + str(undersample) + "_" + strategy
    data, classes = readCsv(inpath, files, True, False)


    # ****** make a  string of what this combination of things is.  Make a lookup for name and method__name__,
    # ****** tmp directory with files 0-24, each file has a line for each method being run.  Should be roughly
    for i in range(len(files)):
        print(files[i])
        # if the  folders for the output files do not exist, create them
        methods = []
        makedirs(dirname(absoluteOutPath + "_binary_" + files[i].replace(".csv","") + ".txt"), exist_ok=True)
        makedirs(dirname(absoluteOutPath + "_multiclass_" + files[i].replace(".csv","") + ".txt"),
                    exist_ok=True)
        # make the output files
        tmpFileBinary = open(absoluteOutPath + "_binary_" + files[i].replace(".csv","") + ".txt", 'w')
        tmpFileMultiClass = open(absoluteOutPath + "_multiclass_" + files[i].replace(".csv","") + ".txt", 'w')
        #workingDataX, workingDataY = prepareData(undersample, featureselection, preprocessing, data, i)
        with catch_warnings():
            simplefilter('ignore')
            trainX, trainY, testX, testY = prepareData(data, classes, i, undersample, featureselection, strategy, preprocessing)
            if isinstance(trainX, int) and trainX == -1:
                return

        """
        kf = KFold(n_splits=25, shuffle=True)
        for train_index, test_index in kf.split(workingDataX):
            trainX, testX = [workingDataX[i] for i in train_index], [workingDataX[j] for j in test_index]
            trainY, testY = [workingDataY[i] for i in train_index], [workingDataY[j] for j in test_index]
        """

        if isLocal:
            if preprocessing is not None:
                print(preprocessing, classifiersettings)
                try:
                    deep = MLPClassifier(hidden_layer_sizes=(150), activation='logistic')
                    t1 = time()
                    deep.fit(trainX, trainY)
                    runtime[deep] = time() - t1
                    settingsLookup[deep] = "150neuronsLogActivation"
                    methods.append(deep)
                except:
                    print("FAIL")
                    pass

                if classifiersettings:
                    # Deep Learning2
                    try:
                        deep2 = MLPClassifier(hidden_layer_sizes=(33))
                        t1 = time()
                        deep2.fit(trainX, trainY)
                        runtime[deep2] = time() - t1
                        settingsLookup[deep2] = '33neurons'
                        methods.append(deep2)
                    except:
                        print("FAIL")
                        pass

                    try:
                        # Deep Learning (SO SLOW!!!) 3
                        deep3 = MLPClassifier(hidden_layer_sizes=(7, 7, 7))
                        t1 = time()
                        deep3.fit(trainX, trainY)
                        runtime[deep3] = time() - t1
                        settingsLookup[deep3] = "5layersVariableSize"
                        methods.append(deep3)
                    except:
                        print("FAIL")
                        pass

            # Autosklearn automatically picks a method and parameters
            # TODO - ADD autosklearn to this (automatically picks the algorithm and does everything for you)
            # autolearn = autosklearn.classification.AutoSklearnClassifier()
            # runtime[m.__class__.__name__] = timer(svm.fit(trainX, trainY))
            #settingsLookup[] = "default"

            try:
                # Support Vector Machines
                svm = SVC()
                t1 = time()
                svm.fit(trainX, trainY)
                runtime[svm] = time() - t1
                settingsLookup[svm] = "default"
                methods.append(svm)
            except:
                print("FAIL")
                pass

            try:
                # Gaussian Naive Bayes
                gaussnb = GaussianNB()
                t1 = time()
                gaussnb.fit(trainX, trainY)
                runtime[gaussnb] = time() - t1
                settingsLookup[gaussnb] = "default"
                methods.append(gaussnb)
            except:
                print("FAIL")
                pass

            try:
                # K-Nearest Neighbors
                knn = KNeighborsClassifier()
                t1 = time()
                knn.fit(trainX, trainY)
                runtime[knn] = time() - t1
                settingsLookup[knn] = "default"
                methods.append(knn)
            except:
                print("FAIL")
                pass

            try:
                # Random Forest
                randfor = RandomForestClassifier()
                t1 = time()
                randfor.fit(trainX, trainY)
                runtime[randfor] = time() - t1
                settingsLookup[randfor] = "default"
                methods.append(randfor)
            except:
                print("FAIL")
                pass

            try:
                # AdaBoost
                ada = AdaBoostClassifier()
                t1 = time()
                ada.fit(trainX, trainY)
                runtime[ada] = time() - t1
                settingsLookup[ada] = "default"
                methods.append(ada)
            except:
                print("FAIL")
                pass

            # Neural Network with Backpropogation
            """
            nntmp = MLPClassifier()
            t1 = time()
            parameters = {'activation': ('logistic', 'relu'), 'alpha': [10.0 ** - numpy.arrange(1, 7)]}
            nn = GridSearchCV(nntmp, parameters)
            nn.fit(trainX, trainY)
            runtime[nn] = time() - t1
            settingsLookup[nn] = "default"
            """

            try:
                # LASSO Regression
                lassocv = LassoCV()
                t1 = time()
                lassocv.fit(trainX, trainY)
                runtime[lassocv] = time() - t1
                settingsLookup[lassocv] = "default"
                methods.append(lassocv)
            except:
                print("FAIL")
                pass

            try:
                # Elastic Net
                elasticnet = ElasticNet()
                t1 = time()
                elasticnet.fit(trainX, trainY)
                runtime[elasticnet] = time() - t1
                settingsLookup[elasticnet] = "default"
                methods.append(elasticnet)
            except:
                print("FAIL")
                pass

            # One-Rule Classifier
            # TODO - no sklearn implementation
            #t1 = time()
            #svm.fit(trainX, trainY)
            #runtime[deep] = time() - t1
            #settingsLookup[] = "default"

            try:
                # Logistic Regression
                logistic = LogisticRegression()
                t1 = time()
                logistic.fit(trainX, trainY)
                runtime[logistic] = time() - t1
                settingsLookup[logistic] = "default"
                methods.append(logistic)
            except:
                print("FAIL")
                pass

            # MARS
            #t1 = time()
            # TODO - not part of sklearn, but allegedly there exists a git repo with an extension to do MARS
            #svm.fit(trainX, trainY)
            #runtime[deep] = time() - t1
            #settingsLookup[] = "default"

            try:
                # PCA
                pca = PCA()
                t1 = time()
                pca.fit(trainX, trainY)
                runtime[pca] = time() - t1
                settingsLookup[pca] = "default"
                methods.append(pca)
            except:
                print("FAIL")
                pass

            try:
                # LDA
                lda = LinearDiscriminantAnalysis()
                t1 = time()
                lda.fit(trainX, trainY)
                runtime[lda] = time() - t1
                settingsLookup[lda] = "default"
                methods.append(lda)
            except:
                print("FAIL")
                pass

            try:
                # One-Class SVM Outlier Detection
                oneclass = OneClassSVM()
                t1 = time()
                oneclass.fit(trainX)
                runtime[oneclass] = time() - t1
                settingsLookup[oneclass] = "default"
                methods.append(oneclass)
            except:
                print("FAIL")
                pass

            # PCA Anomaly Detection
            # TODO - not part of sklearn
            #t1 = time()
            #svm.fit(trainX, trainY)
            #runtime[deep] = time() - t1
            #settingsLookup[] = "default"

            try:
                # CART - sklearn implementation of DecisionTreeClassifier is an optimized version of CART
                cart = DecisionTreeClassifier()
                t1 = time()
                cart.fit(trainX, trainY)
                runtime[cart] = time() - t1
                settingsLookup[cart] = "default"
                methods.append(cart)
            except:
                print("FAIL")
                pass

            if classifiersettings:
                try:
                    # Support Vector Machines 2
                    svm2 = SVC(kernel='linear')
                    t1 = time()
                    svm2.fit(trainX, trainY)
                    runtime[svm2] = time() - t1
                    settingsLookup[svm2] = "linearKernal"
                    methods.append(svm2)
                except:
                    print("FAIL")
                    pass

                """Cannot have negative values
                # Multinomial Naive Bayes
                multinomnb = MultinomialNB()
                t1 = time()
                multinomnb.fit(trainX, trainY)
                runtime[multinomnb] = time() - t1
                settingsLookup[multinomnb] = "default"
                """

                try:
                    # K-Nearest Neighbors 2 - 5NN is standard, this is
                    knn2 = KNeighborsClassifier(weights='distance')
                    t1 = time()
                    knn2.fit(trainX, trainY)
                    runtime[knn2] = time() - t1
                    settingsLookup[knn2] = "dist-weight"
                    methods.append(knn2)
                except:
                    print("FAIL")
                    pass

                try:
                    # Random Forest
                    randfor2 = RandomForestClassifier(criterion='entropy')
                    t1 = time()
                    randfor2.fit(trainX, trainY)
                    runtime[randfor2] = time() - t1
                    settingsLookup[randfor2] = "entropyImp"
                    methods.append(randfor2)
                except:
                    pass

                try:
                    # AdaBoost
                    ada2 = AdaBoostClassifier(base_estimator=GaussianNB)
                    t1 = time()
                    ada2.fit(trainX, trainY)
                    runtime[ada2] = time() - t1
                    settingsLookup[ada2] = "NBbase"
                    methods.append(ada2)
                except:
                    pass

                """
                # Neural Network with Backpropogation
                nntmp2 = MLPClassifier(hidden_layer_sizes=(50,50))
                parameters = {'activation': ('logistic', 'relu'), 'alpha': [10.0 ** - numpy.arrange(1, 7)]}
                t1 = time()
                nn2 = GridSearchCV(nntmp2, parameters)
                nn2.fit(trainX, trainY)
                runtime[nn2] = time() - t1
                settingsLookup[nn2] = "twoLayers"
                """

                try:
                    # Orthogonal Matching Pursuit (linear)
                    ompcv = OrthogonalMatchingPursuitCV()
                    t1 = time()
                    ompcv.fit(trainX, trainY)
                    runtime[ompcv] = time() - t1
                    settingsLookup[ompcv] = "default"
                    methods.append(ompcv)
                except:
                    pass

                try:
                    # Elastic Net
                    multielasticnet = MultiTaskElasticNet()
                    t1 = time()
                    multielasticnet.fit(trainX, trainY)
                    runtime[multielasticnet] = time() - t1
                    settingsLookup[multielasticnet] = "default"
                    methods.append(multielasticnet)
                except:
                    pass

                # One-Rule Classifier 2
                # TODO - no sklearn implementation
                #t1 = time()
                #svm.fit(trainX, trainY)
                #runtime[deep] = time() - t1
                #settingsLookup[] = "default"

                try:
                    # Logistic Regression 2
                    logistic2 = LogisticRegression(solver='sag')
                    t1 = time()
                    logistic2.fit(trainX, trainY)
                    runtime[logistic2] = time() - t1
                    settingsLookup[logistic2] = "SAGsolver"
                    methods.append(logistic2)
                except:
                    pass

                # MARS 2
                # TODO - not part of sklearn, but allegedly there exists a git repo with an extension to do MARS
                #t1 = time()
                #svm.fit(trainX, trainY)
                #runtime[deep] = time() - t1
                #settingsLookup[] = "default"

                try:
                    # PCA 2
                    pca2 = PCA(n_components='mle', svd_solver='full')
                    t1 = time()
                    pca2.fit(trainX, trainY)
                    runtime[pca2] = time() - t1
                    settingsLookup[pca2] = "mleFull"
                    methods.append(pca2)
                except:
                    pass

                try:
                    # LDA 2
                    lda2 = LinearDiscriminantAnalysis(solver='lsqr',shrinkage='auto')
                    t1 = time()
                    lda2.fit(trainX, trainY)
                    runtime[lda2] = time() - t1
                    settingsLookup[lda2] = "LstSqShrink"
                    methods.append(lda2)
                except:
                    pass

                try:
                    # One-Class SVM Outlier Detection 2
                    oneclass2 = OneClassSVM(kernel='linear')
                    t1 = time()
                    oneclass2.fit(trainX)
                    runtime[oneclass2] = time() - t1
                    settingsLookup[oneclass2] = "linearKernel"
                    methods.append(oneclass2)
                except:
                    pass

                # PCA Anomaly Detection 2
                # TODO - not part of sklearn
                #t1 = time()
                #svm.fit(trainX, trainY)
                #runtime[deep] = time() - t1
                #settingsLookup[] = "default"

                try:
                    # CART - sklearn implementation of DecisionTreeClassifier is an optimized version of CART 2
                    cart2 = DecisionTreeClassifier(criterion='entropy')
                    t1 = time()
                    cart2.fit(trainX, trainY)
                    runtime[cart2] = time() - t1
                    settingsLookup[cart2] = "entropyImp"
                    methods.append(cart2)
                except:
                    pass

                try:
                    # Support Vector Machines 3
                    svm3 = SVC(kernel='poly')
                    t1 = time()
                    svm3.fit(trainX, trainY)
                    runtime[svm3] = time() - t1
                    settingsLookup[svm3] = "trinomialKernal"
                    methods.append(svm3)
                except:
                    pass

                try:
                    # Gaussian Naive Bayes 3
                    gaussprocess = GaussianProcessClassifier()
                    t1 = time()
                    gaussprocess.fit(trainX, trainY)
                    runtime[gaussprocess] = time() - t1
                    settingsLookup[gaussprocess] = "default"
                    methods.append(gaussprocess)
                except:
                    pass

                try:
                    # K-Nearest Neighbors
                    knn3 = KNeighborsClassifier(n_neighbors=3)
                    t1 = time()
                    knn3.fit(trainX, trainY)
                    runtime[knn3] = time() - t1
                    settingsLookup[knn3] = "3NN"
                    methods.append(knn3)
                except:
                    pass

                try:
                    # Random Forest 3
                    randfor3 = RandomForestClassifier(max_features=0.5)
                    t1 = time()
                    randfor3.fit(trainX, trainY)
                    runtime[randfor3] = time() - t1
                    settingsLookup[randfor3] = "0.5MaxFeatures"
                    methods.append(randfor3)
                except:
                    pass

                try:
                    # AdaBoost 3
                    ada3 = AdaBoostClassifier(base_estimator=KNeighborsClassifier)
                    t1 = time()
                    ada3.fit(trainX, trainY)
                    runtime[ada3] = time() - t1
                    settingsLookup[ada3] = "KNNbase"
                    methods.append(ada3)
                except:
                    pass

                """
                # Neural Network with Backpropogation 3
                nntmp3 = MLPClassifier(hidden_layer_sizes=15)
                t1 = time()
                parameters = {'activation': ('logistic', 'relu'), 'alpha': [10.0 ** - numpy.arrange(1, 7)]}
                nn3 = GridSearchCV(nntmp3, parameters)
                nn3.fit(trainX, trainY)
                runtime[nn3] = time() - t1
                settingsLookup[nn3] = "15neurons"
                """

                try:
                    # Multi-class LASSO
                    multilasso = MultiTaskLasso()
                    t1 = time()
                    multilasso.fit(trainX, trainY)
                    runtime[multilasso] = time() - t1
                    settingsLookup[multilasso] = "default"
                    methods.append(multilasso)
                except:
                    pass

                try:
                    # *** Bayes Ridge ***
                    bayesridge = BayesianRidge()
                    t1 = time()
                    bayesridge.fit(trainX, trainY)
                    runtime[bayesridge] = time() - t1
                    settingsLookup[bayesridge] = "default"
                    methods.append(bayesridge)
                except:
                    pass

                # One-Rule Classifier 3
                # TODO - no sklearn implementation
                #t1 = time()
                #svm.fit(trainX, trainY)
                #runtime[deep] = time() - t1
                #settingsLookup[] = "default"

                try:
                    # *** SGD ***
                    sgd = SGDClassifier()
                    t1 = time()
                    sgd.fit(trainX, trainY)
                    runtime[sgd] = time() - t1
                    settingsLookup[sgd] = "default"
                    methods.append(sgd)
                except:
                    pass

                # MARS 3
                # TODO - not part of sklearn, but allegedly there exists a git repo with an extension to do MARS
                #t1 = time()
                #svm.fit(trainX, trainY)
                #runtime[deep] = time() - t1
                #settingsLookup[] = "default"

                try:
                    # PCA 3
                    pca3 = PCA(whiten=True)
                    t1 = time()
                    pca3.fit(trainX, trainY)
                    runtime[pca3] = time() - t1
                    settingsLookup[pca3] = "whiten"
                    methods.append(pca3)
                except:
                    pass

                try:
                    # Quadratic LDA
                    qda = QuadraticDiscriminantAnalysis()
                    t1 = time()
                    qda.fit(trainX, trainY)
                    runtime[qda] = time() - t1
                    settingsLookup[qda] = "default"
                    methods.append(qda)
                except:
                    pass

                try:
                    # One-Class SVM Outlier Detection 3
                    oneclass3 = OneClassSVM(kernel='poly')
                    t1 = time()
                    oneclass3.fit(trainX)
                    runtime[oneclass3] = time() - t1
                    settingsLookup[oneclass3] = "trinomialKernel"
                    # TODO - Figure out how scoring works for this one since it only outputs (-1 or +1)
                    methods.append(oneclass3)
                except:
                    pass

                # PCA Anomaly Detection 3
                # TODO - not part of sklearn
                #t1 = time()
                #pc.fit(trainX, trainY)
                #runtime[deep] = time() - t1
                #settingsLookup[] = "default"

                try:
                    # CART - sklearn implementation of DecisionTreeClassifier is an optimized version of CART 3
                    cart3 = DecisionTreeClassifier(splitter='random')
                    t1 = time()
                    cart3.fit(trainX, trainY)
                    runtime[cart3] = time() - t1
                    settingsLookup[cart3] = "randomSplit"
                    methods.append(cart3)
                except:
                    pass
        else:
            if preprocessing is not None:
                print(preprocessing, classifiersettings)
                try:
                    deep = MLPClassifier(hidden_layer_sizes=(150), activation='logistic')
                    t1 = time()
                    deep.fit(trainX, trainY)
                    runtime[deep] = time() - t1
                    settingsLookup[deep] = "150neuronsLogActivation"
                    methods.append(deep)
                except:
                    pass

                if classifiersettings:

                    # Deep Learning2 (SO SLOW!!!)
                    try:
                        deep2 = MLPClassifier(hidden_layer_sizes=(33))
                        t1 = time()
                        deep2.fit(trainX, trainY)
                        runtime[deep2] = time() - t1
                        settingsLookup[deep2] = '33neurons'
                        methods.append(deep2)
                    except:
                        pass

                    try:
                        # Deep Learning (SO SLOW!!!) 3
                        deep3 = MLPClassifier(hidden_layer_sizes=(7, 7, 7))
                        t1 = time()
                        deep3.fit(trainX, trainY)
                        runtime[deep3] = time() - t1
                        settingsLookup[deep3] = "5layersVariableSize"
                        methods.append(deep3)
                    except:
                        pass

            # Autosklearn automatically picks a method and parameters
            # TODO - ADD autosklearn to this (automatically picks the algorithm and does everything for you)
            # autolearn = autosklearn.classification.AutoSklearnClassifier()
            # runtime[m.__class__.__name__] = timer(svm.fit(trainX, trainY))
            # settingsLookup[] = "default"

            try:
                # Support Vector Machines
                svm = SVC()
                t1 = time()
                svm.fit(trainX, trainY)
                runtime[svm] = time() - t1
                settingsLookup[svm] = "default"
                methods.append(svm)
            except:
                pass

            try:
                # Gaussian Naive Bayes
                gaussnb = GaussianNB()
                t1 = time()
                gaussnb.fit(trainX, trainY)
                runtime[gaussnb] = time() - t1
                settingsLookup[gaussnb] = "default"
                methods.append(gaussnb)
            except:
                pass

            try:
                # K-Nearest Neighbors
                knn = KNeighborsClassifier()
                t1 = time()
                knn.fit(trainX, trainY)
                runtime[knn] = time() - t1
                settingsLookup[knn] = "default"
                methods.append(knn)
            except:
                pass

            try:
                # Random Forest
                randfor = RandomForestClassifier()
                t1 = time()
                randfor.fit(trainX, trainY)
                runtime[randfor] = time() - t1
                settingsLookup[randfor] = "default"
                methods.append(randfor)
            except:
                pass

            try:
                # AdaBoost
                ada = AdaBoostClassifier()
                t1 = time()
                ada.fit(trainX, trainY)
                runtime[ada] = time() - t1
                settingsLookup[ada] = "default"
                methods.append(ada)
            except:
                pass

            # Neural Network with Backpropogation
            """
            nntmp = MLPClassifier()
            t1 = time()
            parameters = {'activation': ('logistic', 'relu'), 'alpha': [10.0 ** - numpy.arrange(1, 7)]}
            nn = GridSearchCV(nntmp, parameters)
            nn.fit(trainX, trainY)
            runtime[nn] = time() - t1
            settingsLookup[nn] = "default"
            """

            try:
                # LASSO Regression
                lassocv = LassoCV()
                t1 = time()
                lassocv.fit(trainX, trainY)
                runtime[lassocv] = time() - t1
                settingsLookup[lassocv] = "default"
                methods.append(lassocv)
            except:
                pass

            try:
                # Elastic Net
                elasticnet = ElasticNet()
                t1 = time()
                elasticnet.fit(trainX, trainY)
                runtime[elasticnet] = time() - t1
                settingsLookup[elasticnet] = "default"
                methods.append(elasticnet)
            except:
                pass

            # One-Rule Classifier
            # TODO - no sklearn implementation
            # t1 = time()
            # svm.fit(trainX, trainY)
            # runtime[deep] = time() - t1
            # settingsLookup[] = "default"

            try:
                # Logistic Regression
                logistic = LogisticRegression()
                t1 = time()
                logistic.fit(trainX, trainY)
                runtime[logistic] = time() - t1
                settingsLookup[logistic] = "default"
                methods.append(logistic)
            except:
                pass

            # MARS
            # t1 = time()
            # TODO - not part of sklearn, but allegedly there exists a git repo with an extension to do MARS
            # svm.fit(trainX, trainY)
            # runtime[deep] = time() - t1
            # settingsLookup[] = "default"

            try:
                # PCA
                pca = PCA()
                t1 = time()
                pca.fit(trainX, trainY)
                runtime[pca] = time() - t1
                settingsLookup[pca] = "default"
                methods.append(pca)
            except:
                pass

            try:
                # LDA
                lda = LinearDiscriminantAnalysis()
                t1 = time()
                lda.fit(trainX, trainY)
                runtime[lda] = time() - t1
                settingsLookup[lda] = "default"
                methods.append(lda)
            except:
                pass

            try:
                # One-Class SVM Outlier Detection
                oneclass = OneClassSVM()
                t1 = time()
                oneclass.fit(trainX)
                runtime[oneclass] = time() - t1
                settingsLookup[oneclass] = "default"
                methods.append(oneclass)
            except:
                pass

            # PCA Anomaly Detection
            # TODO - not part of sklearn
            # t1 = time()
            # svm.fit(trainX, trainY)
            # runtime[deep] = time() - t1
            # settingsLookup[] = "default"

            try:
                # CART - sklearn implementation of DecisionTreeClassifier is an optimized version of CART
                cart = DecisionTreeClassifier()
                t1 = time()
                cart.fit(trainX, trainY)
                runtime[cart] = time() - t1
                settingsLookup[cart] = "default"
                methods.append(cart)
            except:
                pass

            if classifiersettings:
                try:
                    # Support Vector Machines 2
                    svm2 = SVC(kernel='linear')
                    t1 = time()
                    svm2.fit(trainX, trainY)
                    runtime[svm2] = time() - t1
                    settingsLookup[svm2] = "linearKernal"
                    methods.append(svm2)
                except:
                    pass

                """Cannot have negative values
                # Multinomial Naive Bayes
                multinomnb = MultinomialNB()
                t1 = time()
                multinomnb.fit(trainX, trainY)
                runtime[multinomnb] = time() - t1
                settingsLookup[multinomnb] = "default"
                """

                try:
                    # K-Nearest Neighbors 2 - 5NN is standard, this is
                    knn2 = KNeighborsClassifier(weights='distance')
                    t1 = time()
                    knn2.fit(trainX, trainY)
                    runtime[knn2] = time() - t1
                    settingsLookup[knn2] = "dist-weight"
                    methods.append(knn2)
                except:
                    pass

                try:
                    # Random Forest
                    randfor2 = RandomForestClassifier(criterion='entropy')
                    t1 = time()
                    randfor2.fit(trainX, trainY)
                    runtime[randfor2] = time() - t1
                    settingsLookup[randfor2] = "entropyImp"
                    methods.append(randfor2)
                except:
                    pass

                try:
                    # AdaBoost
                    ada2 = AdaBoostClassifier(base_estimator=GaussianNB)
                    t1 = time()
                    ada2.fit(trainX, trainY)
                    runtime[ada2] = time() - t1
                    settingsLookup[ada2] = "NBbase"
                    methods.append(ada2)
                except:
                    pass

                """
                # Neural Network with Backpropogation
                nntmp2 = MLPClassifier(hidden_layer_sizes=(50,50))
                parameters = {'activation': ('logistic', 'relu'), 'alpha': [10.0 ** - numpy.arrange(1, 7)]}
                t1 = time()
                nn2 = GridSearchCV(nntmp2, parameters)
                nn2.fit(trainX, trainY)
                runtime[nn2] = time() - t1
                settingsLookup[nn2] = "twoLayers"
                """

                try:
                    # Orthogonal Matching Pursuit (linear)
                    ompcv = OrthogonalMatchingPursuitCV()
                    t1 = time()
                    ompcv.fit(trainX, trainY)
                    runtime[ompcv] = time() - t1
                    settingsLookup[ompcv] = "default"
                    methods.append(ompcv)
                except:
                    pass

                try:
                    # Elastic Net
                    multielasticnet = MultiTaskElasticNet()
                    t1 = time()
                    multielasticnet.fit(trainX, trainY)
                    runtime[multielasticnet] = time() - t1
                    settingsLookup[multielasticnet] = "default"
                    methods.append(multielasticnet)
                except:
                    pass

                # One-Rule Classifier 2
                # TODO - no sklearn implementation
                # t1 = time()
                # svm.fit(trainX, trainY)
                # runtime[deep] = time() - t1
                # settingsLookup[] = "default"

                try:
                    # Logistic Regression 2
                    logistic2 = LogisticRegression(solver='sag')
                    t1 = time()
                    logistic2.fit(trainX, trainY)
                    runtime[logistic2] = time() - t1
                    settingsLookup[logistic2] = "SAGsolver"
                    methods.append(logistic2)
                except:
                    pass

                # MARS 2
                # TODO - not part of sklearn, but allegedly there exists a git repo with an extension to do MARS
                # t1 = time()
                # svm.fit(trainX, trainY)
                # runtime[deep] = time() - t1
                # settingsLookup[] = "default"

                try:
                    # PCA 2
                    pca2 = PCA(n_components='mle', svd_solver='full')
                    t1 = time()
                    pca2.fit(trainX, trainY)
                    runtime[pca2] = time() - t1
                    settingsLookup[pca2] = "mleFull"
                    methods.append(pca2)
                except:
                    pass

                try:
                    # LDA 2
                    lda2 = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
                    t1 = time()
                    lda2.fit(trainX, trainY)
                    runtime[lda2] = time() - t1
                    settingsLookup[lda2] = "LstSqShrink"
                    methods.append(lda2)
                except:
                    pass

                try:
                    # One-Class SVM Outlier Detection 2
                    oneclass2 = OneClassSVM(kernel='linear')
                    t1 = time()
                    oneclass2.fit(trainX)
                    runtime[oneclass2] = time() - t1
                    settingsLookup[oneclass2] = "linearKernel"
                    methods.append(oneclass2)
                except:
                    pass

                # PCA Anomaly Detection 2
                # TODO - not part of sklearn
                # t1 = time()
                # svm.fit(trainX, trainY)
                # runtime[deep] = time() - t1
                # settingsLookup[] = "default"

                try:
                    # CART - sklearn implementation of DecisionTreeClassifier is an optimized version of CART 2
                    cart2 = DecisionTreeClassifier(criterion='entropy')
                    t1 = time()
                    cart2.fit(trainX, trainY)
                    runtime[cart2] = time() - t1
                    settingsLookup[cart2] = "entropyImp"
                    methods.append(cart2)
                except:
                    pass

                try:
                    # Support Vector Machines 3
                    svm3 = SVC(kernel='poly')
                    t1 = time()
                    svm3.fit(trainX, trainY)
                    runtime[svm3] = time() - t1
                    settingsLookup[svm3] = "trinomialKernal"
                    methods.append(svm3)
                except:
                    pass

                try:
                    # Gaussian Naive Bayes 3
                    gaussprocess = GaussianProcessClassifier()
                    t1 = time()
                    gaussprocess.fit(trainX, trainY)
                    runtime[gaussprocess] = time() - t1
                    settingsLookup[gaussprocess] = "default"
                    methods.append(gaussprocess)
                except:
                    pass

                try:
                    # K-Nearest Neighbors
                    knn3 = KNeighborsClassifier(n_neighbors=3)
                    t1 = time()
                    knn3.fit(trainX, trainY)
                    runtime[knn3] = time() - t1
                    settingsLookup[knn3] = "3NN"
                    methods.append(knn3)
                except:
                    pass

                try:
                    # Random Forest 3
                    randfor3 = RandomForestClassifier(max_features=0.5)
                    t1 = time()
                    randfor3.fit(trainX, trainY)
                    runtime[randfor3] = time() - t1
                    settingsLookup[randfor3] = "0.5MaxFeatures"
                    methods.append(randfor3)
                except:
                    pass

                try:
                    # AdaBoost 3
                    ada3 = AdaBoostClassifier(base_estimator=KNeighborsClassifier)
                    t1 = time()
                    ada3.fit(trainX, trainY)
                    runtime[ada3] = time() - t1
                    settingsLookup[ada3] = "KNNbase"
                    methods.append(ada3)
                except:
                    pass

                """
                # Neural Network with Backpropogation 3
                nntmp3 = MLPClassifier(hidden_layer_sizes=15)
                t1 = time()
                parameters = {'activation': ('logistic', 'relu'), 'alpha': [10.0 ** - numpy.arrange(1, 7)]}
                nn3 = GridSearchCV(nntmp3, parameters)
                nn3.fit(trainX, trainY)
                runtime[nn3] = time() - t1
                settingsLookup[nn3] = "15neurons"
                """

                try:
                    # Multi-class LASSO
                    multilasso = MultiTaskLasso()
                    t1 = time()
                    multilasso.fit(trainX, trainY)
                    runtime[multilasso] = time() - t1
                    settingsLookup[multilasso] = "default"
                    methods.append(multilasso)
                except:
                    pass

                try:
                    # *** Bayes Ridge ***
                    bayesridge = BayesianRidge()
                    t1 = time()
                    bayesridge.fit(trainX, trainY)
                    runtime[bayesridge] = time() - t1
                    settingsLookup[bayesridge] = "default"
                    methods.append(bayesridge)
                except:
                    pass

                # One-Rule Classifier 3
                # TODO - no sklearn implementation
                # t1 = time()
                # svm.fit(trainX, trainY)
                # runtime[deep] = time() - t1
                # settingsLookup[] = "default"

                try:
                    # *** SGD ***
                    sgd = SGDClassifier()
                    t1 = time()
                    sgd.fit(trainX, trainY)
                    runtime[sgd] = time() - t1
                    settingsLookup[sgd] = "default"
                    methods.append(sgd)
                except:
                    pass

                # MARS 3
                # TODO - not part of sklearn, but allegedly there exists a git repo with an extension to do MARS
                # t1 = time()
                # svm.fit(trainX, trainY)
                # runtime[deep] = time() - t1
                # settingsLookup[] = "default"

                try:
                    # PCA 3
                    pca3 = PCA(whiten=True)
                    t1 = time()
                    pca3.fit(trainX, trainY)
                    runtime[pca3] = time() - t1
                    settingsLookup[pca3] = "whiten"
                    methods.append(pca3)
                except:
                    pass

                try:
                    # Quadratic LDA
                    qda = QuadraticDiscriminantAnalysis()
                    t1 = time()
                    qda.fit(trainX, trainY)
                    runtime[qda] = time() - t1
                    settingsLookup[qda] = "default"
                    methods.append(qda)
                except:
                    pass

                try:
                    # One-Class SVM Outlier Detection 3
                    oneclass3 = OneClassSVM(kernel='poly')
                    t1 = time()
                    oneclass3.fit(trainX)
                    runtime[oneclass3] = time() - t1
                    settingsLookup[oneclass3] = "trinomialKernel"
                    methods.append(oneclass3)
                except:
                    pass

                # PCA Anomaly Detection 3
                # TODO - not part of sklearn
                # t1 = time()
                # pc.fit(trainX, trainY)
                # runtime[deep] = time() - t1
                # settingsLookup[] = "default"

                try:
                    # CART - sklearn implementation of DecisionTreeClassifier is an optimized version of CART 3
                    cart3 = DecisionTreeClassifier(splitter='random')
                    t1 = time()
                    cart3.fit(trainX, trainY)
                    runtime[cart3] = time() - t1
                    settingsLookup[cart3] = "randomSplit"
                    methods.append(cart3)
                except:
                    pass


        # TODO - add ensemble of these to the mix?
        # ****** RECALL ****** with the files I'm using, none of them are that big...20,000 total at worst
        """
        if classifiersettings: # add autolearn when it is working

            methods = [deepLearn, deepLearn2, deepLearn3, svm, svm2, svm3, gaussnb, multinomialnb, gaussnb3,
                       knn, knn3, knn5, knn10, randfor, randfor2, randfor3, ada, ada2, ada3, nn, nn2, nn3,
                       lassocv, lassocv2, lassocv3, logistic, logistic2, logistic3, ols, ols2, ols3,
                       elasticnet, elasticnet2, elasticnet3, onerule, onerule2, onerule3, mars, mars2, mars3,
                       pca, pca2, pca3, lda, lda2, lda3, oneclass, oneclass2, oneclass3, pcaanomaly,
                       pcaanomaly2, pcaanomaly3, cart, cart2, cart3]

            # add ellipse envelope and isolationForest
            methods = [svm, svm2, svm3, gaussnb, knn, knn2, knn3, randfor, randfor2, randfor3,
                       ada, ada3, lassocv, logistic, logistic2,
                       elasticnet, pca, pca2, pca3, lda, lda2, oneclass, oneclass2, oneclass3, cart,
                       cart2, cart3, ompcv, gaussprocess, multilasso, bayesridge, sgd, qda ]
        else: # add autolearn when it is working

            methods = [something deep learning, svm, gaussnb, knn, randfor, ada, nn, lassocv, elasticnet,
                       onerule, logistic, mars, pca, lda, oneclass, pcaanomaly, cart]

            methods = [svm, gaussnb, knn, randfor, ada, lassocv, elasticnet, logistic, pca, lda, oneclass, cart]

        """


        results = {m: [] for m in methods}

        # FORMAT OF : MC_kBest_0.25_bellweather_ knn_1_local_standardize
        #   dataset_featSel_underSamp_bellweather vs round robin_classifier_classifier settings_local vs global_preprocessing option

        # data print (binary) is of format : name tn-fp-fn-tp-runtime
        loops = len(testX) if isinstance(testX[0], (list, tuple)) else 1
        for k in range(loops):
            for m in methods:
                tester = [[0, 0], [0, 0]]
                if m.__class__.__name__ != 'OneClassSVM':
                    print(m.__class__.__name__)
                    try:
                        if strategy == 'bellweather':
                            tester = confusion_matrix(testY[k], m.predict(testX[k]))
                        else:
                            tester = confusion_matrix(testY, m.predict(testX))
                    except:
                        pass
                else:
                    # make real positives and real negatives
                    pos, neg = [], []
                    if strategy =='bellweather':
                        pos = [testX[k][x] for x in range(len(testX[k])) if testY[k][x] != 0]
                        neg = [testX[k][x] for x in range(len(testX[k])) if testY[k][x] == 0]
                    else:
                        pos = [testX[x] for x in range(len(testX)) if testY[x] != 0]
                        neg = [testX[x] for x in range(len(testX)) if testY[x] == 0]

                    if len(pos) > 0:
                        pos_pred = m.predict(pos)
                    if len(neg) > 0:
                        neg_pred = m.predict(neg)
                    n_error_pos = len([1 for x in pos_pred if x == 1])
                    n_error_neg = len([1 for x in neg_pred if x == -1])
                    tester = [[len(pos)-n_error_pos, n_error_pos],[n_error_neg, len(neg)-n_error_neg]]
                if numpy.sum(numpy.sum(tester)) != 0:
                    tn = tester[0][0]
                    fn = numpy.sum(tester, axis=0)[0] - tester[0][0]
                    fp = numpy.sum(tester, axis=1)[0] - tester[0][0]
                    tp = numpy.sum(numpy.sum(tester)) - sum([tn, fn, fp])
                    tmpFileBinary.write("_".join([methodDetails, m.__class__.__name__, settingsLookup[m], "local" if isLocal else "global", preprocessing if preprocessing is not None else "none", " "]))
                    tmpFileBinary.write(str(tn) + "-" + str(fp) + "-" + str(fn) + "-" + str(tp)+ "-" + str(runtime[m]) + "\n")

                    if len(tester) > 2:
                        tmpFileMultiClass.write("_".join(
                            [methodDetails, m.__class__.__name__, settingsLookup[m], "local" if isLocal else "global",
                             preprocessing if preprocessing is not None else "none", " "]))
                        writeString = ','.join(['-'.join([str(i) for i in tester[j]]) for j in range(len(tester))])
                        tmpFileMultiClass.write(str(len(tester)) + " " + writeString + str(runtime[m]) + "\n")
        tmpFileBinary.close()
        tmpFileMultiClass.close()


        # TODO create all the output files, output the output(after each run, but keep all and overwrite so that the file gets progressively closer to complete)
        # TODO - MAKE AN ANALYSIS PROGRAM
    postprocessing()

def postprocessing():
    # TODO - This whole method
    print("you know, in a perfect world postprocessing would go here")



if __name__ != '__main__':
    """
    file1 = ['accumulo.csv', 'bookkeeper.csv', 'camel.csv', 'cassandra.csv', 'cxf.csv', 'derby.csv', 'felix.csv',
             'hive.csv', 'openjpa.csv', 'pig.csv', 'wicket.csv']
    file2 = ['ant2.csv', 'arc2.csv', 'berek2.csv', 'camel2.csv', 'elearning2.csv', 'ivy2.csv','jedit2.csv', 'log4j2.csv', 'lucene2.csv', 'poi2.csv', 'synapse2.csv', 'xerces2.csv']
    file3 = ['cm1.csv', 'pc1.csv', 'pc3.csv', 'pc4.csv']
    """
    file1 = ['accumulo.csv', 'bookkeeper.csv', 'camel.csv', 'cassandra.csv', 'cxf.csv', 'derby.csv']
    file2 = ['ant2.csv', 'arc2.csv', 'berek2.csv']
    file3 = ['cm1.csv', 'pc1.csv', 'pc3.csv']
    outpath = 'C:\\Users\\Andrew\\Documents\\Schools\\Grad School\\NCSU - Comp Sci\\Research\\TransferLearning\\'
    inpath1 = '..\\spatialtree\\Mining Datasets\\Bellweather\\'
    inpath2 = '..\\spatialtree\\Mining Datasets\\Bellweather\\Promise Datasets\\CK2\\'
    inpath3 = '..\\spatialtree\\Mining Datasets\\Bellweather\\Promise Datasets\\MC\\'

    #inputSets = [(file1, inpath1, 'PM'), (file2, inpath2, 'CK'), (file3, inpath3, 'MC')]
    inputSets = [(file2, inpath2, 'CK'), (file3, inpath3, 'MC')]
    wholeOrSingly = ['bellweather', 'roundrobin']  # roundrobin you train on n-1 and test on 1, bellweather train on 1 and test each of the other n-1 individually
    undersamplingOptions = [None, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 5.0, 10.0] # add SMOTE, OTHER1, OTHER2, THINK OF OTHER UNDERSAMPLING TECHNIQUESx3]
    preprocessing = ['standardize', 'normalize'] # add None later, is going too slowly.  think of other things to add
    #**************# where to add "remove duplicates"??
    classifierSettings = [True] # True runs a bunch of different settings, False runs only the basic ones
    local = [False] # still need to run all the local versions, but I haven't decided what I mean by "local" yet
    featureSelections = [None, 0.1, 0.3, 0.5, 0.7, 0.9, 'kBest', 'autoSelect']

    for i in inputSets:
        for ws in wholeOrSingly:
            for fs in featureSelections:
                for us in undersamplingOptions:
                    for lo in local:
                        for pr in preprocessing:
                            for cs in classifierSettings:
                                print(i[0], i[2], i[1], outpath, us, lo, ws, pr, cs, fs)
                                with catch_warnings():
                                    simplefilter("ignore")
                                    transferLearning(i[0], i[2], i[1], outpath, us, lo, ws, pr, cs, fs)



# MAKE METHOD
file1 = ['accumulo.csv']
inpath1 = '..\\spatialtree\\Mining Datasets\\Bellweather\\'
X, y = readCsv(inpath1, file1)
X, y = X[0], y[0]
results = {}
meth = []
print("Data read in")
print(len(X), len(y), len(X[0]))
kf = KFold(n_splits=2, shuffle=True)
for train_index, test_index in kf.split(X):
    trainX, testX = [X[i] for i in train_index], [X[j] for j in test_index]
    trainY, testY = [1 if y[i] > 0.5 else 0  for i in train_index], [y[j] for j in test_index]
    print("train-test split complete")
    trainX = StandardScaler().fit_transform(trainX)
    # Deep Learning2 (SO SLOW!!!)
    parameters = {'alpha' : 10.0 ** -numpy.arange(1, 7), 'activation': ['relu', 'tanh', 'logistic']}
    print(parameters)
    deep = MLPClassifier()

    grid = GridSearchCV(deep, parameters, error_score=0, scoring=make_scorer(f1_score))
    grid.fit(trainX, trainY)
    print(grid.best_params_)
    print(grid.best_estimator_)
    exit()
    deep2 = MLPClassifier(grid.best_params_).fit(trainX, trainY)

    #deep2 = OneClassSVM().fit(trainX)

    print(deep2)
    tester = confusion_matrix(testY, deep2.predict(testX))
    print(tester)
    exit()
    #results[deep2] = "success"
    print("MLP trained")
    print(deep2.__class__.__name__)
    meth.append(deep2)
    for m in meth:
        print(m)
        print(results[m])
        print(m.__class__.__name__)
    exit()
    tester = confusion_matrix(testY, deep2.predict(testX))
    print(tester)
    tn = tester[0][0]
    fn = numpy.sum(tester, axis=0)[0] - tester[0][0]
    fp = numpy.sum(tester, axis=1)[0] - tester[0][0]
    tp = sum(sum(tester)) - sum([tn, fn, fp])
    print(str(tn) + "-" + str(fp) + "-" + str(fn) + "-" + str(tp))
    # GET OUTPUT AND FIGURE OUT HOW TO MAKE IT SHOW UP RIGHT
