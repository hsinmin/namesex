# -*- coding: utf-8 -*-
#import numpy as np
import pkg_resources
import csv


#load data from the csv file
class testdata:
    def __init__(self):
        filename = pkg_resources.resource_filename('namesex', 'data/testdata.csv')
        f = open(filename, 'r', newline='', encoding = 'utf8')
        mydata = csv.DictReader(f)
        sexlist = []
        namelist = []
        for arow in mydata:
            sexlist.append(int(arow['sex'].strip()))
            gname = arow['name'].strip()
            namelist.append(gname)

        self.gname = namelist
        self.sex = sexlist


if __name__ == "__main__":
    import namesex
    testdata = testdata()
    nsl = namesex.namesex()
    pred = nsl.predict(testdata.gname)
    print("The first 5 given names are: {}".format(testdata.gname[0:5]))
    print("    and their sex:          {}".format(testdata.sex[0:5]))
    print("    and their predicted sex:{}".format(pred[0:5]))
    accuracy = np.sum(pred == testdata.sex) / len(pred)
    print(" Prediction accuracy = {}".format(accuracy))
