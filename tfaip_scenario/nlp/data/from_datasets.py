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
from dataclasses import field, dataclass
from typing import Type, Iterable, Optional

import datasets
from paiargparse import pai_dataclass, pai_meta

from tfaip import DataGeneratorParams, PipelineMode, Sample, TrainerPipelineParamsBase
from tfaip.data.pipeline.datagenerator import DataGenerator


@pai_dataclass
@dataclass
class FromDatasetsDataGeneratorParams(DataGeneratorParams):
    dataset_name: str = field(
        default="germeval_14", metadata=pai_meta(help="The dataset to select (chose also fashion_mnist).")
    )

    dataset_config_name: Optional[str] = field(
        default=None, metadata=pai_meta(help="The configuration name of the dataset to use (via the datasets library).")
    )
    eval_on_test_list: Optional[bool] = field(
        default=False, metadata=pai_meta(help="Set True for evaluation on test list")
    )

    @staticmethod
    def cls() -> Type["DataGenerator"]:
        return FromDatasetsDataGenerator


class FromDatasetsDataGenerator(DataGenerator[FromDatasetsDataGeneratorParams]):
    def __init__(self, mode: PipelineMode, params: "FromDatasetsDataGeneratorParams"):
        super().__init__(mode, params)
        dataset = datasets.load_dataset(params.dataset_name, params.dataset_config_name)
        # self.dataset_info = datasets.get_dataset_infos(params.dataset_name)
        # tag_list = self.dataset_info[params.dataset_name].features['ner_tags'].feature.names
        # with tempfile.TemporaryDirectory() as temp_dir:
        #     with open(os.path.join(temp_dir, f"tags_{params.dataset_name}.txt"), "w") as tags_fp:
        #         tags_fp.write("\n".join(tag_list))
        #         print(f"tempfile: {tags_fp.name}")
        #         pass

        if mode == PipelineMode.TRAINING:
            data = dataset["train"]
        elif params.eval_on_test_list:
            data = dataset["test"]
        else:
            data = dataset["validation"]
        self.data = data

    @staticmethod
    def to_samples(data) -> Iterable[Sample]:
        for sample_ in data:
            sample = Sample(inputs={"text": sample_["tokens"]}, targets={"tag_ids": sample_["ner_tags"]})
            yield sample

    def __len__(self):
        return len(self.data)

    def generate(self) -> Iterable[Sample]:
        return self.to_samples(self.data)


@pai_dataclass
@dataclass
class FromDatasetsTrainerGeneratorParams(
    TrainerPipelineParamsBase[FromDatasetsDataGeneratorParams, FromDatasetsDataGeneratorParams]
):
    train_val: FromDatasetsDataGeneratorParams = field(
        default_factory=FromDatasetsDataGeneratorParams, metadata=pai_meta(mode="flat")
    )

    def train_gen(self) -> FromDatasetsDataGeneratorParams:
        return self.train_val

    def val_gen(self) -> Optional[FromDatasetsDataGeneratorParams]:
        return self.train_val
