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

import ast
import json
import logging
import math
import os
from dataclasses import dataclass, field
from typing import Dict, Union, Optional, List

from dataclasses_json import config
from paiargparse import pai_dataclass

from tfaip.util.typing import enforce_types
from tfaip_addons.util.file.stringmapper import StringMapper


def flatten(list_list: List[List[object]]) -> List[object]:
    return [item for sublist in list_list for item in sublist]


@enforce_types
@pai_dataclass
@dataclass
class Base(object):
    id: Optional[str]
    # disabled: Optional[bool]


@enforce_types
@pai_dataclass
@dataclass
class Dimension:
    w: int
    h: int

    def __repr__(self):
        return f"{self.w},{self.h}"


@enforce_types
@pai_dataclass
@dataclass
class WithConfidence(object):
    confidence: float = 0.0


@enforce_types
@pai_dataclass
@dataclass
class StringConfidenced(WithConfidence):
    content: Optional[str] = field(default="")


@enforce_types
@pai_dataclass
@dataclass
class FloatConfidenced(WithConfidence):
    content: float = 0.0


@enforce_types
@pai_dataclass
@dataclass
class ResultClassification:
    label: str
    score: float = 0.0
    isValid: bool = False


@enforce_types
@pai_dataclass
@dataclass
class Classification:
    results: List[ResultClassification] = field(default_factory=list)
    moduleId: str = field(default="")
    moduleLabel: str = field(default="")
    threshold: float = field(default=0.0)


import numpy as np


@enforce_types
@pai_dataclass
@dataclass
class Rectangle(object):
    x: int
    y: int
    w: int
    h: int

    def __repr__(self):
        return f"{self.x},{self.y},{self.w},{self.h}"

    @property
    def xmin(self):
        return self.x

    @property
    def xmax(self):
        return self.x + self.w - 1

    @property
    def ymin(self):
        return self.y

    @property
    def ymax(self):
        return self.y + self.h - 1

    def get_as_polygon(self):
        xs = [
            round(self.x),
            round(self.x),
            round(self.x + self.w),
            round(self.x + self.w),
        ]
        ys = [
            round(self.y),
            round(self.y + self.h),
            round(self.y + self.h),
            round(self.y),
        ]
        return Polygon(xs=xs, ys=ys)


@enforce_types
@pai_dataclass
@dataclass
class Polygon:
    xs: List[int]
    ys: List[int]

    def get_bounds(self) -> Rectangle:
        x = np.asarray(self.xs)
        y = np.asarray(self.ys)
        xmin = int(np.min(x))
        xmax = int(np.max(x))
        ymin = int(np.min(y))
        ymax = int(np.max(y))
        # TODO: @Tobi: okay?
        return Rectangle(x=xmin, y=ymin, w=xmax - xmin + 1, h=ymax - ymin + 1)

    def __repr__(self):
        return ";".join([f"{x},{y}" for x, y in zip(self.xs, self.ys)])


@enforce_types
@pai_dataclass
@dataclass
class Box(object):
    """
     NOTE: anchor point is the lower left corner based on the text orientation
    (think of a straight baseline with height). This differs to the classical CV conventions.
    """

    x: int
    y: int
    w: int
    h: int
    angle: float = field(default=0.0)

    def __repr__(self):
        return f"{self.x},{self.y},{self.w},{self.h},{self.angle}"

    def get_rect(self, start=0.0, end=1.0) -> Rectangle:
        assert 0.0 <= start <= 1.0
        assert 0.0 <= end <= 1.0
        poly = self.get_as_polygon(start=start, end=end)
        return poly.get_bounds()

    def get_as_polygon(self, start=0.0, end=1.0) -> Polygon:
        assert 0.0 <= start <= 1.0
        assert 0.0 <= end <= 1.0
        xs = [
            round(self.x + start * self.w * math.cos(self.angle)),
            round(self.x + end * self.w * math.cos(self.angle)),
            round(self.x + end * self.w * math.cos(self.angle) + self.h * math.sin(self.angle)),
            round(self.x + start * self.w * math.cos(self.angle) + self.h * math.sin(self.angle)),
        ]
        ys = [
            round(self.y + start * self.w * math.sin(self.angle)),
            round(self.y + end * self.w * math.sin(self.angle)),
            round(self.y + end * self.w * math.sin(self.angle) - self.h * math.cos(self.angle)),
            round(self.y + start * self.w * math.sin(self.angle) - self.h * math.cos(self.angle)),
        ]
        return Polygon(xs=xs, ys=ys)

    @property
    def xmin(self):
        return self.get_rect().xmin

    @property
    def xmax(self):
        return self.get_rect().xmax

    @property
    def ymin(self):
        return self.get_rect().ymin

    @property
    def ymax(self):
        return self.get_rect().ymax


def encode_box(box: Optional[Box]) -> Optional[str]:
    if box is None:
        return None
    return box.__repr__()


def decode_box(string: Optional[str]) -> Optional[Box]:
    if not string:
        return None
    s = string.split(",")
    if len(s) != 5:
        raise Exception(f"cannot split string '{string}' into 5 parts, obtain {len(s)}.")
    return Box(int(s[0]), int(s[1]), int(s[2]), int(s[3]), float(s[4]))


def encode_dimension(dims: Optional[Dimension]) -> Optional[str]:
    if dims is None:
        return None
    return dims.__repr__()


def decode_dimension(r: Optional[str]) -> Optional[Dimension]:
    if r is None:
        return None
    return _str_to_dimensions(r)


def encode_polygon(polygon: Union[Polygon, Dict]) -> Optional[str]:
    if polygon is None:
        return None
    if isinstance(polygon, Polygon):
        return polygon.__repr__()
    return ";".join([f"{x},{y}" for x, y in zip(polygon["xs"], polygon["ys"])])


def decode_polygon(r: Optional[str]) -> Optional[Polygon]:
    if r is None:
        return None
    return _str_to_polygon(r)


@enforce_types
@pai_dataclass
@dataclass
class PolygonConfidenced(WithConfidence):
    content: Polygon = field(default=None, metadata=config(decoder=decode_polygon, encoder=encode_polygon))


@enforce_types
@pai_dataclass
@dataclass
class PolygonConfidenced(WithConfidence):
    content: Polygon = field(default=None, metadata=config(decoder=decode_polygon, encoder=encode_polygon))


@enforce_types
@pai_dataclass
@dataclass
class IndexRange:
    begin: int = 0
    end: int = 0
    totalLength = 0


@enforce_types
@pai_dataclass
@dataclass
class ResultEntitySnippet:
    pageId: Optional[str] = ""
    lineId: Optional[str] = field(default="")
    rangeText: Optional[IndexRange] = field(default=None)
    rangeCm: Optional[IndexRange] = field(default=None)
    pos: Optional[Box] = field(default_factory=list, metadata=config(decoder=decode_box, encoder=encode_box))
    coordinates: float = 0.0
    text: str = ""


@enforce_types
@pai_dataclass
@dataclass
class ResultEntity:
    text: str = ""
    label: str = ""
    id: Optional[str] = ""
    score: float = 0.0
    coordinates: Optional[Polygon] = field(
        default=None, metadata=config(decoder=decode_polygon, encoder=encode_polygon)
    )
    subEntities: List[ResultEntitySnippet] = field(default_factory=list)


@enforce_types
@pai_dataclass
@dataclass
class ModulResultEntity:
    results: List[ResultEntity] = field(default_factory=list)
    moduleId: str = field(default="")
    moduleLabel: str = field(default="")
    threshold: float = field(default=0.0)


@enforce_types
@pai_dataclass
@dataclass
class Word(Base):
    id: Optional[str] = None
    text: Optional[StringConfidenced] = field(default=None)
    wordboxBounds: Optional[Box] = field(default_factory=list, metadata=config(decoder=decode_box, encoder=encode_box))
    wordboxMainBody: Optional[Box] = field(
        default_factory=list, metadata=config(decoder=decode_box, encoder=encode_box)
    )
    properties: Dict[str, str] = field(default_factory=dict)

    def get_text(self) -> Union[None, str]:
        return self.text.content if self.text is not None else None

    def get_box(self) -> Union[None, Box]:
        return self.wordboxBounds

    def get_mainbody(self) -> Union[None, Box]:
        return self.wordboxMainBody


@enforce_types
@pai_dataclass
@dataclass
class Line(Base):
    coordinates: Optional[Polygon] = field(
        default=None, metadata=config(decoder=decode_polygon, encoder=encode_polygon)
    )
    baseline: Optional[PolygonConfidenced] = field(default=None)
    properties: Optional[Dict[str, str]] = field(default_factory=dict)
    words: Optional[List[Word]] = field(default_factory=list)
    text: Optional[StringConfidenced] = field(default=None)


@enforce_types
@pai_dataclass
@dataclass
class Region:
    id: Optional[str] = None
    coordinates: Optional[Polygon] = field(
        default=None, metadata=config(decoder=decode_polygon, encoder=encode_polygon)
    )
    lines: List[Line] = field(default_factory=list)

    properties: Dict[str, str] = field(default_factory=dict)
    classifications: Dict[str, Classification] = field(default_factory=dict)
    regions: List["Region"] = field(default_factory=list)

    def get_subregions(self) -> List["Region"]:
        flat = self.regions or []
        res = []
        for r in flat:
            res.append(r)
            res.extend(r.get_subregions())
        return res

    def get_lines(self) -> List[Line]:
        return sum([region.lines for region in self.get_regions()], []) + self.lines

    def get_words(self) -> List[Word]:
        return sum([line.words for line in self.get_lines()], [])


@enforce_types
@pai_dataclass
@dataclass
class Page(Region):
    classifications: Dict[str, Classification] = field(default_factory=dict)
    imageDimsProcess: Optional[Dimension] = field(
        default=None, metadata=config(decoder=lambda x: decode_dimension(x), encoder=lambda x: encode_dimension(x))
    )
    imageDimsOriginal: Optional[Dimension] = field(
        default=None, metadata=config(decoder=lambda x: decode_dimension(x), encoder=lambda x: encode_dimension(x))
    )
    angleMod90: Optional[FloatConfidenced] = field(default=None)

    # TODO (tobi) parse reading order
    # readingOrder: Optional[List[Classification]] = field(default_factory=list)

    def get_regions(self) -> List[Region]:
        res = []
        for region in self.regions:
            res.append(region)
            res.extend(region.get_subregions())
        return res


def _skip_default(thedict: Dict[str, object]) -> Dict[str, object]:
    outdict = {}
    for key in thedict.keys():
        val = thedict[key]
        if val is None:
            pass
        elif isinstance(val, (int, float, str, list, dict, set)) and not val:  # not val means empty or default
            pass
        elif isinstance(val, dict):
            outdict[key] = _skip_default(val)
        elif isinstance(val, list):
            outdict[key] = [_skip_default(v) for v in val]
        elif isinstance(val, set):
            outdict[key] = {_skip_default(v) for v in val}
        else:
            outdict[key] = val
    return outdict


def _rm_cls(thedict: Dict[str, object]) -> None:
    if "__cls__" in thedict:
        del thedict["__cls__"]
    for v in thedict.values():
        if isinstance(v, dict):
            _rm_cls(v)
        elif isinstance(v, list):
            [_rm_cls(e) for e in v]
        elif isinstance(v, set):
            [_rm_cls(e) for e in v]


@enforce_types
@pai_dataclass
@dataclass
class File(Base):
    srcImgPath: str
    properties: Dict[str, str] = field(default_factory=dict)
    classifications: Dict[str, Classification] = field(default_factory=dict)
    entities: Dict[str, ModulResultEntity] = field(default_factory=dict)
    pages: List[Page] = field(default_factory=list)

    def get_pages(self) -> List[Page]:
        return self.pages

    def get_regions(self, allow_multiple_pages=False) -> List[Region]:
        if not self.pages:
            return []
        if len(self.pages) > 1 and not allow_multiple_pages:
            raise Exception(
                f"cannot return regions if multiple pages (={len(self.pages)}) are available."
                + " Either iterate over pages or set parameter 'allow_multiple_pages'=True."
            )
        return sum([page.get_regions() for page in self.get_pages()], [])

    def get_lines(self, allow_multiple_pages=False) -> List[Line]:
        if not self.pages:
            return []
        if len(self.pages) > 1 and not allow_multiple_pages:
            raise Exception(
                f"cannot return lines if multiple pages (={len(self.pages)}) are available."
                + " Either iterate over pages or set parameter 'allow_multiple_pages'=True."
            )
        return sum([region.lines for region in self.get_regions()], [])

    def get_words(self, allow_multiple_pages=False) -> List[Word]:
        if not self.pages:
            return []
        if len(self.pages) > 1 and not allow_multiple_pages:
            raise Exception(
                f"cannot return words if multiple pages (={len(self.pages)}) are available."
                + " Either iterate over pages or set parameter 'allow_multiple_pages'=True."
            )
        return sum([line.words for line in self.get_lines()], [])

    def get_classifications(self) -> Dict[str, Classification]:
        return self.classifications

    def get_entities(self) -> Dict[str, ModulResultEntity]:
        return self.entities

    @staticmethod
    def load(path_to_file) -> "File":
        if not os.path.exists(path_to_file):
            logging.warning(f"File not found: {path_to_file}")
            return None
        with open(path_to_file, "r") as fp:
            return File.from_json(fp.read(), infer_missing=True)

    def save(self, path_to_file, skip_defaults=False, skip_cls=True) -> None:
        with open(path_to_file, "w") as fp:
            saveable_dict = self.to_dict()
            if skip_defaults:
                saveable_dict = _skip_default(saveable_dict)
            if skip_cls:
                _rm_cls(saveable_dict)
            json.dump(saveable_dict, fp, indent=2)


def _str_to_polygon(str_from_json: str) -> Polygon:
    if str_from_json[0] == "[":
        reg_proc = ast.literal_eval(str_from_json.replace(";", ","))
        xs = [coord[0] for coord in reg_proc]
        ys = [coord[1] for coord in reg_proc]
    else:
        points_as_str = [e.split(",") for e in str_from_json.strip().split(";")]
        xs = [int(e[0]) for e in points_as_str]
        ys = [int(e[1]) for e in points_as_str]
    return Polygon(xs, ys)


def _str_to_dimensions(str_from_json: str) -> Dimension:
    wh = str_from_json.split(",")
    return Dimension(int(wh[0]), int(wh[1]))


def _get_class_list(
    classifications: Dict[str, Classification],
    category_mapper: StringMapper,
    threshold=0.0,
    with_classification_id=True,
):
    res = []
    for key, classification in classifications.items():
        class_id = f"{key}:" if with_classification_id else ""
        for result in classification.results:
            lbl = result.label
            if float(result.score) < threshold:
                logging.info(f"ignore class {lbl} since score {result.score} > {threshold}")
                continue
            id = _get_index_or_name(f"{class_id}{lbl}", category_mapper)
            if id is not None:
                res.append(id)
    return res


def _get_index_or_name(label: str, category_mapper: Optional[StringMapper]) -> Optional[Union[int, str]]:
    if category_mapper is None:
        return label
    id = category_mapper.get_channel(label)
    if id == category_mapper.get_oov_id():
        logging.info(f"ignore class {label} since it is not in the category-mapper")
        return None
    return id


def get_lines_from_region(region: Region):
    ret = []
    ret.extend(region.lines)
    for subregion in region.regions:
        ret.extend(get_lines_from_region(subregion))
    return ret


def get_lines_from_page(page: Page):
    ret = []
    ret.extend(page.lines)
    for subregion in page.regions:
        ret.extend(get_lines_from_region(subregion))
    return ret
