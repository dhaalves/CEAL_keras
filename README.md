## Information
Cost-Effective Active Learning (CEAL) for Deep Image Classification Implementation with keras

Model - Resnet18v2

Dataset - Cifar10

## Running
```sh
python CEAL_keras.py
```

### Parameters
```sh
  -h, --help            show this help message and exit
  -verbose VERBOSE      Verbosity mode. 0 = silent, 1 = progress bar, 2 = one
                        line per epoch. default: 0
  -epochs EPOCHS        Number of epoch to train. default: 5
  -batch_size BATCH_SIZE
                        Number of samples per gradient update. default: 32
  -chkt_filename CHKT_FILENAME
                        Model Checkpoint filename to save
  -t FINE_TUNNING_INTERVAL, --fine_tunning_interval FINE_TUNNING_INTERVAL
                        Fine-tuning interval. default: 1
  -T MAXIMUM_ITERATIONS, --maximum_iterations MAXIMUM_ITERATIONS
                        Maximum iteration number. default: 10
  -i INITIAL_ANNOTATED_PERC, --initial_annotated_perc INITIAL_ANNOTATED_PERC
                        Initial Annotated Samples Percentage. default: 0.1
  -dr THRESHOLD_DECAY, --threshold_decay THRESHOLD_DECAY
                        Threshold decay rate. default: 0.0033
  -delta DELTA          High confidence samples selection threshold. default:
                        0.05
  -K UNCERTAIN_SAMPLES_SIZE, --uncertain_samples_size UNCERTAIN_SAMPLES_SIZE
                        Uncertain samples selection size. default: 2000
  -uc UNCERTAIN_CRITERIA, --uncertain_criteria UNCERTAIN_CRITERIA
                        Uncertain selection Criteria: 'lc' (Least Confidence),
                        'ms' (Margin Sampling), 'en' (Entropy). default: lc
  -ce COST_EFFECTIVE, --cost_effective COST_EFFECTIVE
                        whether to use Cost Effective high confidence sample
                        pseudo-labeling. default: True
```
