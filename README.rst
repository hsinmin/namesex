namesex
-------

Namesex is a package that predicts the gender tendency of a Chinese given name. This module comes with two prediction models trained on 10,730 Chinese given names (in traditional Chinese) with reliable gender lables collected from public data. The first prediction model is a random forest classifier that can be invoked by predict(). This model takes three types of features: the given name, the unigram of given name, and a vector of size one hundred extracted from a skip-gram word-to-vector model trained separately using a news corpus collected from tw.yahoo.com. This news corpus contains 87,848,812 Chinese characters.

The second prediction model is a L2 regularized logistic regression that can be invoked by predict_logic(). This model uses the given names and the unigrams of given names only. Both prediction methods take a list of names and output predicted gender tendency (1 for male and 0 for female) or probability of being a male name.

While gensim was used to train the skip-gram word2vec model, this project does not depend on gensim because the trained model was extracted to a dictionary structure for the convenient use of this project.  This project, nonetheless, depends on numpy, scipy, and sklearn. Windows users may want to install numpy, scipy, and sklearn using pre-compiled binary packages before installing namesex via pip. If you just want something that "just work" and does not want to install sklearn, consider using the sister project, namesex_light, that depends only on numpy. Namesex_light provides the same preduction function using a regularized logistic regression trained on the same dataset. Namesex_light should be faster than the predict() here. The prediction accuracy of namesex_light, however, is lower than the predict() function in namesex.

Additional information about namesex and namesex_light can be found `in another document (in Chinese) <https://github.com/hsinmin/namesex/blob/master/vignettee_namesex_exp1.ipynb>`_.

The prediction performance of the random forest and logistic regression models evaluated by ten-fold cross validation is listed below.


Random Forest
=============
========= =========== =====================
Metric    Performance Performance Std. Dev.
--------- ----------- ---------------------
Accuracy  0.9486      0.007072
F1        0.9470      0.007963
Precision 0.9525      0.008399
Recall    0.9417      0.012985
Logloss   161.54      4.101283
========= =========== =====================

L2 Regularized Logistic Regression
==================================

========= =========== =====================
Metric    Performance Performance Std. Dev.
--------- ----------- ---------------------
Accuracy  0.8957      0.007327
F1        0.8920      0.007873
Precision 0.8852      0.012238
Recall    0.8991      0.008936
Logloss   114.35      6.413972
========= =========== =====================

The random forest model clearly has a higher accuracy and F1 score. We have also tested the k-nearest-neighbor (KNN) model (not reported here). KNN and logistic regression have a similar level of performance, and was excluded for obvious reasons.

Use pip/pip3 to install namesex.::

    pip install namesex

To use namesex, pass in an array or list of given names to predict(). For each element in the input list, predict() returns 1 or 0 for male or female prediction. Set "predprob = True" to return probability of being a male name. The following is a simple sample code.::

    >>> import namesex
    >>> ns = namesex.namesex()
    >>> ns.predict(['民豪', '愛麗', '志明'])
    array([1, 0, 1])
    >>> ns.predict(['民豪', '愛麗', '志明'], predprob=True)
    array([0.8245    , 0.25695238, 0.85      ])

Note that namesex was trained using Chinese given names only. However, it may be used to classifier translated names as well::

    >>> ns.predict(['莎拉波娃', '阿波羅', '雷', '艾美', '布蘭妮', '瑪麗亞'])
    array([0, 1, 1, 0, 0, 0])

The model was trained using given names only. As a result, for the best performance, the input data should be preprocessed to keep given names only.::

    >>> ns.predict(['黃志明春嬌', '黃志明', '志明', '黃春嬌', '春嬌'], predprob = True)
    array([0.61825   , 0.79039286, 0.85      , 0.3646    , 0.3716    ])

In the above example, the family name has a minor effect on the prediction. Concatenating a male and female name somehow neutralize (toward 0.5) the gender tendency.

Testing Dataset
---------------

This package comes with a small testing dataset that was not used for model training. The following sample code illustrate a simple usage.::

    >>> testdata = namesex.testdata()
    >>> ns = namesex.namesex()
    >>> pred = ns.predict(testdata.gname)
    >>> pred2 = ns.predict_logic(testdata.gname)
    >>> import numpy as np
    >>> accuracy = np.mean(pred == testdata.sex)
    >>> print(" Prediction accuracy (random forest) = {}".format(accuracy))
     Prediction accuracy (random forest) = 0.8921568627450981
    >>> accuracy2 = np.mean(pred2 == testdata.sex)
    >>> print(" Prediction accuracy (logistic reg) = {}".format(accuracy2))
     Prediction accuracy (logistic reg) = 0.8627450980392157


For both methods, the accuracy is slightly lower compared to the accuracy of ten-fold cross valudation. Random forest is still better compared to logistic regression.


Model Training
--------------

The module come with the training data. It is possible to train the model by yourself.
