# tf2_neiss_nlp
scenarios for natural language processing
it contains several pretraining tasks for bert models and fine-tuning for named entity recognition (ner).

# Setup
### Install tf2_neiss_nlp:
`pip install -U pip setuptools`

`pip install  <path_to:tf2_neiss_nlp>`

### Install tf2_neiss_nlp as a developer:
Linking the cloned git repo into the virutal environment. 
First consider cloning and install [tfaip](https://github.com/Planet-AI-GmbH/tf2_aip_base) in development mode too. 
Otherwise, `tfaip` will be installed from PyPi.

`pip install -e <path_to:tf2_neiss_nlp>`

Run tests:
`cd tf2_neiss_nlp`
`python -m unittest discover tfaip_scenario_test`

# Usage
see [tf2_aip_base-wiki](https://github.com/Planet-AI-GmbH/tf2_aip_base/wiki) for general information.

_Contributions are welcome, and they are greatly appreciated!_
