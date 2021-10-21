# Copyright 2021 The neiss authors. All Rights Reserved.
#
# This file is part of tf_neiss_nlp.
#
# tf_neiss_nlp is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by the
# Free Software Foundation, either version 3 of the License, or (at your
# option) any later version.
#
# tf_neiss_nlp is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
# or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for
# more details.
#
# You should have received a copy of the GNU General Public License along with
# tf_neiss_nlp. If not, see http://www.gnu.org/licenses/.
# ==============================================================================
import os

from setuptools import setup, find_packages

this_dir = os.path.dirname(os.path.realpath(__file__))
# Parse version
main_ns = {}
__version__ = "1.2.6"

setup(
    name="tfneissnlp",
    version=__version__,
    packages=find_packages(exclude=["test/*"]),
    license="GPL-v3.0",
    long_description=open(os.path.join(this_dir, "README.md")).read(),
    long_description_content_type="text/markdown",
    include_package_data=True,
    author="ProjectNeiss",
    author_email="jochen.zoellner@uni-rostock.de",
    url="https://github.com/NEISSproject/tf2_neiss_nlp",
    download_url="https://github.com/NEISSproject/tf2_neiss_nlp/archive/{}.tar.gz".format(__version__),
    entry_points={},
    python_requires=">=3.7",
    install_requires=open("requirements.txt").read().split("\n"),
    data_files=[("", ["requirements.txt"])],
    keywords=[
        "machine learning",
        "tensorflow",
        "framework",
        "natural language processing",
    ],
)
