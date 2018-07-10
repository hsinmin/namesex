# -*- coding: utf-8 -*-

import operator
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn import linear_model
import pkg_resources, pickle, gzip

class namesex:
    def __init__(self, n_jobs=-1, n_estimators = 500, loadmodel = True, \
                   w2v_filename = \
                   pkg_resources.resource_filename('namesex', 'model/w2v_dictvec_sg_s100i20.pickle')):
        self.gname_count = dict()
        self.gnameug_count = dict()
        self.feat_dict = dict()
        self.num_feature = 0
        self.num_gname = 0
        self.lrmodelintcp = None
        self.lrmodelcoef = None
        self.w2v_dictvec = None
        #mean of diverge
        self.w2v_pooling = "diverge"
        self.rfmodel = RandomForestClassifier(n_estimators=n_estimators, n_jobs=n_jobs)

        #load w2v data
        with open(w2v_filename, "rb") as f:
            self.w2v_dictvec = pickle.load(f)
        #firstchar = self.w2v_dictvec.keys()[0]
        firstchar = next(iter(self.w2v_dictvec.keys()))
        self.w2v_vecsize = len(self.w2v_dictvec[firstchar])

        if loadmodel:
            self.load_coef()
            #print("Loading pre-trained random forest model...")
            self.load_rf_model()
            #print("   ....done.")

    def gen_feature_id(self, namevec, min_ugram = 2, min_gname = 2):
        for gname in namevec:
            if gname in self.gname_count:
                self.gname_count[gname] += 1
            else:
                self.gname_count[gname] = 1
            for achar in gname:
                if achar in self.gnameug_count:
                    self.gnameug_count[achar] += 1
                else:
                    self.gnameug_count[achar] = 1

        gname_sorted = sorted(self.gname_count.items(), key=operator.itemgetter(1), reverse=True)
        gnameug_sorted = sorted(self.gnameug_count.items(), key=operator.itemgetter(1), reverse=True)
        #print("Top 20 given names: {}".format(gname_sorted[0:20]))
        #for apair in gname_sorted[0:20]:
        #    print("{}, female-male: {}".format(apair, gname_sex_count[apair[0]]))

        fid = 0
        for apair in gname_sorted:
            if apair[0] in self.feat_dict:
                print("Error! {} already exists".format(apair[0]))
            else:
                if apair[1] >= min_gname:
                    self.feat_dict[apair[0]] = fid
                    fid += 1
        # number of gnames included
        self.num_gname = fid

        for apair in gnameug_sorted:
            if apair[1] >= min_ugram:
                if apair[0] in self.feat_dict:
                    #print("Warning! {} already exists".format(apair[0]))
                    pass
                else:
                    self.feat_dict[apair[0]] = fid
                    fid += 1
        # add "_Other_Value_"
        self.feat_dict["_Other_Value_"] = fid
        self.num_feature = fid + 1

    def feature_coding(self, aname):
        tmpfeat = list()
        has_other = 0
        if aname in self.feat_dict:
            tmpfeat.append(self.feat_dict[aname])

        for achar in aname:
            if achar in self.feat_dict:
                tmpfeat.append(self.feat_dict[achar])

        if len(tmpfeat) == 0:
            tmpfeat.append(self.feat_dict["_Other_Value_"])
        return tmpfeat

    #generate unigram and gname feature array
    def gen_feat_array(self, namevec):
        #name_given_int = list()
        x_array = np.zeros((len(namevec), self.num_feature), "int")
        for id, aname in enumerate(namevec):
            #name_given_int.append(self.feature_coding(aname))
            x_array[id, self.feature_coding(aname)] = 1
        return x_array
    def gen_feat_array_w2v(self, namevec):
        x_train = self.gen_feat_array(namevec)
        xw2v_train1, xw2v_train2 = self.extract_w2v(namevec)

        if self.w2v_pooling == "mean":
            x2_train = np.concatenate((x_train, xw2v_train1), axis=1)
        else:
            x2_train = np.concatenate((x_train, xw2v_train2), axis=1)
        return x2_train

    # add w2v features.
    # w2vmodel1.vector_size
    # note: use global variables.
    def extract_w2v(self, namearray):
        xw2v1 = np.zeros((len(namearray), self.w2v_vecsize), "float")
        xw2v2 = np.zeros((len(namearray), self.w2v_vecsize), "float")
        for i, aname in enumerate(namearray):
            # preserve the mean
            vec_mean = np.zeros((self.w2v_vecsize), "float")
            # want to preserve the part that is farest from zero for each dimension
            # positive part
            vec_diverge1 = np.zeros((self.w2v_vecsize), "float")
            # negative part
            vec_diverge0 = np.zeros((self.w2v_vecsize), "float")
            nc1 = 0
            for achar in aname:
                try:
                    # tmp = w2vmodel1[achar]
                    tmp = self.w2v_dictvec[achar]
                    tmp = tmp / np.linalg.norm(tmp)
                    vec_mean = vec_mean + tmp
                    nc1 += 1
                    # divergent
                    ind1 = tmp >= 0
                    tmp1 = tmp * ind1
                    tmp0 = tmp * (1 - ind1)
                    vec_diverge1 = np.max(np.vstack((vec_diverge1, tmp1)), axis=0)
                    vec_diverge0 = np.min(np.vstack((vec_diverge0, tmp0)), axis=0)
                except:
                    #print("{} not in w2v model, skip".format(achar))
                    pass

                if nc1 > 1:
                    vec_mean = vec_mean / nc1
                vec_diverge = vec_diverge1 + vec_diverge0
                xw2v1[i] = vec_mean
                xw2v2[i] = vec_diverge
        return xw2v1, xw2v2

    def train(self, namevec, sexvec, c2 = 10):
        #tran logistic regression (unigram and given names) and random forest (with word2vec features)
        #print("Training random forest")

        self.gen_feature_id(namevec)
        y_train = np.asarray(sexvec)

        logreg = linear_model.LogisticRegression(C=c2)
        x_array = self.gen_feat_array(namevec)
        logreg.fit(x_array, y_train)
        #self.lrmodel = logreg
        self.lrmodelcoef = np.transpose(logreg.coef_)
        self.lrmodelintcp = logreg.intercept_

        x2_train = self.gen_feat_array_w2v(namevec)
        self.rfmodel.fit(x2_train, y_train)

    def predict(self, namevec, predprob = False):
        #this model will not be updated, at least for now
        if len(self.feat_dict) == 0:
            print("Warning: No feature table! Maybe load_model() first?")
        x2_test = self.gen_feat_array_w2v(namevec)

        #ypred_rf = clf.predict(x2_test)
        ypred_prob_rf = self.rfmodel.predict_proba(x2_test)
        if predprob == True:
            return ypred_prob_rf[:,1]
        else:
            return np.asarray(ypred_prob_rf[:,1] >=0.5, "int")


    def predict_logic(self, namevec, predprob = False):
        #this model will not be updated, at least for now
        if len(self.feat_dict) == 0:
            print("Warning: No feature table! Maybe load_model() first?")
        x_array = self.gen_feat_array(namevec)

        tmp1 = np.matmul(x_array, self.lrmodelcoef) + self.lrmodelintcp
        ypred = 1/(1+np.exp(-tmp1))
        ypred = ypred.squeeze()
        if predprob == True:
            return ypred
        else:
            return np.asarray(ypred >= 0.5, "int")


    def load_coef(self, \
                   logic_filename = \
                   pkg_resources.resource_filename('namesex','model/logic_light.pickle')):
        #import pickle
        with open(logic_filename, "rb") as f:
            model = pickle.load(f)

        self.lrmodelcoef = model["lrmodelcoef"]
        self.lrmodelintcp = model["lrmodelintcp"]
        self.feat_dict = model["feat_dict"]
        self.num_feature = model["num_feature"]



    def load_rf_model(self, filename = pkg_resources.resource_filename("namesex", "model/namesex_rf.bin")):
        #import pickle, gzip
        #with open(filename, "rb") as f:
        #    rf = pickle.load(f)
        with gzip.open(filename, 'rb') as f:
            rf = pickle.load(f)
        self.rfmodel = rf
        #return(nsl)

    #def load_model(self, filename = pkg_resources.resource_filename("namesex", "model/namesex_model.pickle")):
    #    import pickle
    #    with open(filename, "rb") as f:
    #        nsl = pickle.load(f)
    #    return(nsl)

    #def save_model(self, filename = pkg_resources.resource_filename("namesex", "model/namesex_model.pickle")):
    #    import pickle
    #    with open(filename, "wb") as f:
    #        pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)
    #    #with gzip.open(filename, 'wb') as f:
    #    #    pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)

    def export_model(self, coeffilename = pkg_resources.resource_filename("namesex","model/logic_light.pickle"), \
                     rffilename = pkg_resources.resource_filename("namesex", "model/namesex_rf.bin")):
        #import pickle, gzip

        model = dict()
        model["lrmodelcoef"] = self.lrmodelcoef
        model["lrmodelintcp"] = self.lrmodelintcp
        model["feat_dict"] = self.feat_dict
        model["num_feature"] = self.num_feature

        with open(coeffilename, "wb") as f:
            pickle.dump(model, f)

        #with open(filename, "wb") as f:
        #    pickle.dump(self.rfmodel, f, pickle.HIGHEST_PROTOCOL)
        with gzip.open(rffilename, 'wb') as f:
            pickle.dump(self.rfmodel, f, pickle.HIGHEST_PROTOCOL)

if __name__ == "__main__":
    import csv
    from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
    #import pickle

    #mode = "evaluate" # look at performance using a train-test split

    #only this is working for now
    #mode = "explore"
    mode = "train_production" #train a model for production
    mode = "check_ok"

    #load data
    f = open('data/namesex_data_v2.csv', 'r', newline='', encoding = 'utf8')
    mydata = csv.DictReader(f)
    sexlist = []
    namelist = []
    foldlist = []
    for arow in mydata:
        sexlist.append(int(arow['sex'].strip()))
        gname = arow['gname'].strip()
        namelist.append(gname)
        foldlist.append(int(arow['fold'].strip()))

    sexlist = np.asarray(sexlist)
    namelist = np.asarray(namelist)
    foldlist = np.asarray(foldlist)


    if mode == "explore":
        sex_train = sexlist[foldlist != 0]
        sex_test = sexlist[foldlist == 0]

        name_train = namelist[foldlist != 0]
        name_test = namelist[foldlist == 0]

        nsl = namesex()
        #nsl.train(name_train, sex_train, c2=10)
        #nsl.load_coef()
        ypred = nsl.predict_logic(name_test, predprob = True)
        ypred2 = nsl.predict_logic(name_test, predprob=False)

        ind1 = sex_test == ypred2
        acc = np.mean(ind1)
        print("accuracy = ", acc)

        print("Training random forest")
        nsl.train_random_forest(name_train, sex_train)
        print("training completed.")

        ypred3 = nsl.predict(name_test, predprob=False)
        ypred4 = nsl.predict(name_test, predprob=True)
        ind2 = sex_test == ypred3
        acc3 = np.mean(ind2)
        print("accuracy (random forest = ", acc3)

        #testing data
        #import namesex_light
        filename = pkg_resources.resource_filename('namesex', 'data/testdata.csv')
        f = open(filename, 'r', newline='', encoding = 'utf8')
        mydata = csv.DictReader(f)
        sexlist = []
        namelist = []
        for arow in mydata:
            sexlist.append(int(arow['sex'].strip()))
            gname = arow['name'].strip()
            namelist.append(gname)

        #nsl2 = namesex_light.namesex_light()
        pred = nsl.predict_logic(namelist)
        print("The first 5 given names are: {}".format(namelist[0:5]))
        print("    and their sex:          {}".format(sexlist[0:5]))
        print("    and their predicted sex:{}".format(pred[0:5]))
        accuracy = np.sum(pred == sexlist) / len(pred)
        print(" Prediction accuracy = {}".format(accuracy))

        pred2 = nsl.predict(namelist)
        accuracy2 = np.mean(pred2 == sexlist)
        print(" Prediction accuracy (rf) = {}".format(accuracy2))
    elif mode == "train_production":
        np.random.seed(1034)
        ns = namesex(loadmodel=False)
        print("Training models (logistic reg and random forest)")
        ns.train(namelist, sexlist)
        print("training completed.")
        print("save model.")
        #ns.save_model()
        ns.export_model()


        # testing data
        # import namesex_light
        filename = pkg_resources.resource_filename('namesex', 'data/testdata.csv')
        f = open(filename, 'r', newline='', encoding='utf8')
        mydata = csv.DictReader(f)
        testsexlist = []
        testnamelist = []
        for arow in mydata:
            testsexlist.append(int(arow['sex'].strip()))
            gname = arow['name'].strip()
            testnamelist.append(gname)

        pred = ns.predict_logic(testnamelist)
        print("The first 5 given names are: {}".format(testnamelist[0:5]))
        print("    and their sex:          {}".format(testsexlist[0:5]))
        print("    and their predicted sex:{}".format(pred[0:5]))

        pred2 = ns.predict(testnamelist)
        accuracy2 = np.mean(pred2 == testsexlist)
        print(" Prediction accuracy (rf) = {}".format(accuracy2))
        accuracy = np.mean(pred == testsexlist)
        print(" Prediction accuracy = {}".format(accuracy))
    elif mode == "check_ok":
        print("Check saved model.")
        ns = namesex()
        #ns = ns.load_model()
        ns.load_rf_model()


        # import namesex_light
        filename = pkg_resources.resource_filename('namesex', 'data/testdata.csv')
        f = open(filename, 'r', newline='', encoding='utf8')
        mydata = csv.DictReader(f)
        testsexlist = []
        testnamelist = []
        for arow in mydata:
            testsexlist.append(int(arow['sex'].strip()))
            gname = arow['name'].strip()
            testnamelist.append(gname)

        pred = ns.predict_logic(testnamelist)
        print("The first 5 given names are: {}".format(testnamelist[0:5]))
        print("    and their sex:          {}".format(testsexlist[0:5]))
        print("    and their predicted sex:{}".format(pred[0:5]))

        pred2 = ns.predict(testnamelist)
        accuracy2 = np.mean(pred2 == testsexlist)
        print(" Prediction accuracy (rf) = {}".format(accuracy2))
        accuracy = np.mean(pred == testsexlist)
        print(" Prediction accuracy = {}".format(accuracy))
