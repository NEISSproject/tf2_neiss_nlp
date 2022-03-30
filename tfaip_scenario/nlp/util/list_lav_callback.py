import json
import os
from typing import List

from tfaip.lav.callbacks.lav_callback import LAVCallback

from tfaip.lav.lav import LAV
from tfaip.scenario.listfile.params import ListFileTrainerPipelineParams
from tfaip.scenario.scenariobase import ScenarioBase, TScenarioParams


class ListFileScenario(ScenarioBase[TScenarioParams, ListFileTrainerPipelineParams]):
    """
    Base-Class for a Scenario working with list files.
    A list file is a simple text file where each line is the path to a sample.
    The ListFileScenario uses ListFileTrainerPipelineParams which create a DataGenerator that will yield each line
    as a Sample's input and target. This must then be processed by DataProcessors.
    """

    @classmethod
    def lav_cls(cls):
        return ListFileLAV

class ListFileLAV(LAV):
    """Custom LAV to add additional callbacks."""

    def _custom_callbacks(self) -> List[LAVCallback]:
        return [ListFileLAVCallback()]



class ListFileLAVCallback(LAVCallback):
    """Custom LAVCallback used in LAV Scenarios.

    On the end of LAV, this will dump all metrics in the model_path (if `store_results`==True).
    """

    def on_lav_end(self, result):
        if self.lav.params.store_results:
            dump_dict = {
                "metrics": {
                    k: v for k, v in result.items() if (not isinstance(v, bytes) and not type(v).__module__ == "numpy")
                },
                "lav_params": self.lav.params.to_dict(),
                "data_params": self.data.params.to_dict(),
                "model_params": self.model.params.to_dict(),
            }
            json_fn = os.path.join(
                self.lav.params.model_path,
                f"lav_results_{'_'.join([os.path.basename(l) for l in self.current_data_generator_params.lists])}.json",
            )

            with open(json_fn, "w") as json_fp:
                json.dump(dump_dict, json_fp, indent=2)

    def _on_lav_end(self, result):
        if self.lav.params.store_results:
            dump_dict = {
                "metrics": {
                    k: v for k, v in result.items() if (not isinstance(v, bytes) and not type(v).__module__ == "numpy")
                },
                "lav_params": self.lav.params.to_dict(),
                "data_params": self.data.params.to_dict(),
                "model_params": self.model.params.to_dict(),
            }
            json_fn = os.path.join(
                self.lav.params.model_path,
                f"lav_results_{'_'.join([os.path.basename(l) for l in self.current_data_generator_params.lists])}.json",
            )

            with open(json_fn, "w") as json_fp:
                json.dump(dump_dict, json_fp, indent=2)


class ListFileLAV(LAV):
    """Custom LAV to add additional callbacks."""

    def _custom_callbacks(self) -> List[LAVCallback]:
        return [ListFileLAVCallback()]
