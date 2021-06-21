# tf2_neiss_nlp
scenarios for natural language processing
it contains several pretraining tasks for bert models and fine tuning for named entity recognition (ner).

# Setup
### Install tf2_neiss_nlp:

`pip install --use-feature=2020-resolver <path_to:tf2_neiss_nlp>`

### Install tf2_neiss_nlp as a developer:
The cloned git repos is linked to the virtual environment and can be edited.
First consider to clone and install [tfaip](https://github.com/Planet-AI-GmbH/tf2_aip_base) in development mode too. 
Otherwise it will be installed from PyPi.

`pip install -e <path_to:tf2_neiss_nlp>`

And add `<path_to:tf2_neiss_nlp/tfneissnlp>` to the `TF_AIP_SCENARIOS` environment variable:
* run `export TF_AIP_SCENARIOS=<path_to:tf2_neiss_nlp/tfneissnlp>:$TF_AIP_SCENARIOS` 
  in your shell or add it permanent to your virtual env activate script

Run tests:
`cd tf2_neiss_nlp`
`python -m unittest discover test`

Note: for some tests it is necessary to mark the tfaip directory as source to fix "import test.util.training"
# Usage
see [tf2_aip_base-wiki](https://github.com/Planet-AI-GmbH/tf2_aip_base/wiki) for general information.

_Contributions are welcome, and they are greatly appreciated!_
