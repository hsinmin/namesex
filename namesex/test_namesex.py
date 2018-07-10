# -*- coding: utf-8 -*-
from unittest import TestCase

import namesex_light
import numpy as np

class test_namesex_light(TestCase):
    def test_feat_dict_other_value(self):
        s = namesex_light.namesex_light()
        self.assertTrue("_Other_Value_" in s.feat_dict)

    def test_feat_dict_size(self):
        s = namesex_light.namesex_light()
        self.assertTrue(len(s.feat_dict) > 1000)
    def test_predict(self):
        nsl = namesex_light.namesex_light()
        pred1 = nsl.predict(['民豪', '愛麗', '志明'])
        ans1 = np.array([1, 0, 1])

        self.assertTrue(sum(pred1 == ans1) == len(ans1))
