# Copyright 2020 The neiss authors. All Rights Reserved.
#
# This file is part of tf2_neiss_nlp.
#
# tf2_neiss_nlp is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by the
# Free Software Foundation, either version 3 of the License, or (at your
# option) any later version.
#
# tf2_neiss_nlp is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
# or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for
# more details.
#
# You should have received a copy of the GNU General Public License along with
# tf2_neiss_nlp. If not, see http://www.gnu.org/licenses/.
# ==============================================================================
import os
from distutils import sysconfig

from setuptools import setup, find_packages

from tfaip_scenario.nlp import __version__

site_packages_path = os.path.sep.join(sysconfig.get_python_lib().split(os.path.sep)[-3:])
setup(
    name="tfneissnlp",
    version=__version__,
    license="GPL-v3.0",
    long_description=open("README.md").read(),
    packages=find_packages(exclude=["test/*"]),
    long_description_content_type="text/markdown",
    include_package_data=True,
    author="ProjectNeiss",
    author_email="jochen.zoellner@uni-rostock.de",
    url="https://github.com/NEISSproject/tf2_neiss_nlp",
    download_url="https://github.com/NEISSproject/tf2_neiss_nlp/archive/{}.tar.gz".format(__version__),
    entry_points={},
    python_requires=">=3.7",
    install_requires=open("requirements.txt").read().split("\n"),
    keywords=[
        "machine learning",
        "tensorflow",
        "framework",
        "natural language processing",
    ],
)
