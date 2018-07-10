# -*- coding: utf-8 -*-
from setuptools import setup
#from distutils.core import setup

def readme():
    with open('README.rst') as f:
        return f.read()

setup(
  name='namesex',
  packages = ['namesex'],
  version = '0.1.12',
  description='A gender classifier for Chinese given names',
  author = 'Hsin-Min Lu, Yu-Lun Li, Chi-Yu Lin',
  author_email = 'luim@ntu.edu.tw',
  include_package_data = True,
  long_description=readme(),
  long_description_content_type = "/text/x-rst",
  classifiers=[  # Optional
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        'Development Status :: 3 - Alpha',

        # Indicate who your project is intended for
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',

        # Pick your license as you wish
        'License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)',

        # Specify the Python versions you support here. In particular, ensure
        # that you indicate whether you support Python 2, Python 3 or both.
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],
  keywords = ['classify_sex Chinese_given_name'],
  install_requires=['numpy', 'sklearn'],
  zip_safe=False,
  package_data={'':['data/*', 'model/*']},
  test_suite='nose.collector',
  tests_require=['nose'],
  #package_data={  # Optional
  #      'testdata': ['data/testdata.csv'],
  #      'model': ['model/*'],
  #},
  #url = 'https://github.com/yulun0528/namesex',
  #download_url = 'https://github.com/yulun0528/namesex/archive/2.5.tar.gz'
)
