# FLShield
In this repository, code is for our IEEE S&P 2024 paper [FLShield: A Validation Based Federated Learning Framework to Defend Against Poisoning Attacks](https://arxiv.org/pdf/2308.05832.pdf)

## Installation
Recommended python version: 3.8.x, 3.9.x

Create a virtual environment and install the dependencies listed in `requirements.txt`:
```
pip install -r requirements.txt
```

## Usage
The basic usage is as follows:
```
python main.py --aggregation_methods=X --attack_methods=Y --type=Z
```
where `X` is the aggregation method and can take values in 'mean', 'geom_median','flame', 'flshield', 'afa', 'foolsgold'

`Y` is the poisoning attack method and can take values in - 'targeted_label_flip', 'dba', 'inner_product_manipulation', 'attack_of_the_tails', 'semantic_attack'

`Z` is the dataset and can take values in 'emnist', 'fmnist', 'cifar', 'loan'

Note: in order to run FLShield with bijective version, `--bijective_flshield` should be added to the command line.

Committee election behavior (for `flshield`/`fedcsap`):
- `no_models` controls how many client updates are trained/collected per round.
- `committee_size` controls how many validators are elected as committee members.
- If `committee_size` is not set, the code defaults it to `no_models`.

So if you run with `no_models=25` and do not pass `--committee_size`, the elected committee will also have 25 members.
To get 25 training clients with a 5-member committee, pass `--committee_size=5`.

If you want **25 total participants** in the whole federation and **5 committee members**, run for example:
```
python main.py --type=cifar --aggregation_methods=flshield --attack_methods=targeted_label_flip --mal_pcnt=0.2 --resumed_model=false --epochs=210 --bijective_flshield --number_of_total_participants=25 --committee_size=5 --no_models=20
```
In this setup, 5 selected participants are committee validators and up to 20 non-committee clients are trained each round.

Do participants change each round?
- The global participant pool (IDs) is fixed for a run.
- Committee members and training clients are re-sampled each epoch, so the active set usually changes round by round.

Different scenarios require different parameters which is listed in `utils/jinja.yaml`
Some of them are changable from the command line, for example adding `--noniid=one_class_expert` will run the experiment with one class expert data distribution.





## Citation
If you find our work useful in your research, please consider citing:
```
@article{kabir2023flshield,
  title={FLShield: A Validation Based Federated Learning Framework to Defend Against Poisoning Attacks},
  author={Kabir, Ehsanul and Song, Zeyu and Rashid, Md Rafi Ur and Mehnaz, Shagufta},
  journal={arXiv preprint arXiv:2308.05832},
  year={2023}
}
```
## Acknowledgement 
- [AI-secure/DBA](https://github.com/AI-secure/DBA)
- [ebagdasa/backdoor_federated_learning](https://github.com/ebagdasa/backdoor_federated_learning)
- [krishnap25/RFA](https://github.com/krishnap25/RFA)
- [DistributedML/FoolsGold](https://github.com/DistributedML/FoolsGold)
