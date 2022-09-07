# DeepSimulator: Learning Accurate Business Process Simulation Models from Event Logs via Automated Process Discovery and Deep Learning

DeepSimulator is a hybrid approach able to learn process simulation models from event logs wherein a (stochastic) process model is extracted via DDS techniques and then combined with a DL model to generate timestamped event sequences. This code can perform the next tasks:


* Training generative models using an event log as input.
* Generate full event logs using the trained generative models.
* Assess the similarity between the original log and the generated one.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. 

### Prerequisites

To execute this code you just need to install Anaconda or Conda in your system, and create an environment using the *environment.yml* specification provided in the repository.

## Running the script

Once created the environment, you can execute the tool from a terminal specifying the input event log name and any of the following parameters:

* `--file (required)`: event log in XES format, the event log must be previously located in the `input_files/event_logs` folder
* `--update_gen/--no-update_gen (optional, default=False)`: Refers to whether you want to update the sequences generation model previously discovered. If this parameter is added, the entire discovery pipeline will be executed. Additionally, if this parameter is set, the number of repetitions of every experiment can be configured with the parameter `--s_gen_repetitions (optional, default=5)`, and the number of experiments with `--s_gen_max_eval (optional, default=30)`.
* `--update_ia_gen/--no-update_ia_gen (optional, default=False)`: Refers to whether you want to update the inter-arrivals generation model previously discovered. If this parameter is added, the entire discovery pipeline will be executed.
* `--update_times_gen/--no-update_times_gen (optional, default=False)`: Refers to whether you want to update the deep-learning times generation models previously discovered. If this parameter is added, the entire discovery pipeline will be executed. Additionally, if this parameter is set, the number of training epochs can be configured with the parameter `--t_gen_epochs (optional, default=200)`, and the number of experiments with `--t_gen_max_eval (optional, default=12)`.
* `--save_models/--no-save_models (optional, default=True)`: Refers to whether or not you want to save the discovered models.
* `--evaluate/--no-evaluate (optional, default=True)`: Refers to whether or not you want to perform a final assessment of the accuracy of the final simulation model.
* `--mining_alg (optional, default='sm1')`: version of SplitMiner to use. Available options: 'sm1', 'sm2', 'sm3'.

**Example of basic execution:**

```shell
(deep_simulator) PS C:\DeepSimulator> python .\pipeline.py --file Production.xes
```

**Example of execution updating the deep-learning times generation models**

```shell
(deep_simulator) PS C:\DeepSimulator> python .\pipeline.py --file Production.xes --update_times_gen --t_gen_epochs 20 --t_gen_max_eval 3
```

## Examples

The datasets, models, and evaluation results can be found at <a href="https://doi.org/10.5281/zenodo.5734443" target="_blank">Zenodo</a>. The paper of the approach can be found at  <a href="https://doi.org/10.1007/978-3-031-07472-1_4" target="_blank">CAiSE'22 Paper</a>
## Authors

* **Manuel Camargo**
* **Marlon Dumas**
* **Oscar Gonzalez-Rojas**