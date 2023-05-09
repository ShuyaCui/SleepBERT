## run pretrain model
```
python pretrain.py --template train_bert --dataset_code UKB --test False --val True --validateafter 1 --bin_num 10
```
## Test a pretrained checkpoint
```
python validate.py --template validate_bert --test False --val True --bin_num 10
```
* The checkpoint file is stored at /experiments/test_2022-04-26_0/models/best_acc_model.pth. You may need to download the checkpoint file manually from git LFS.

## Acknowledgements
Training pipeline is implemented based on this repo https://github.com/hw-du/CBiT. We would like to thank the contributors for their work.