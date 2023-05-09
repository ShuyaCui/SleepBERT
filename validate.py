from options import args
from models import BERTModel
from datasets import dataset_factory
from dataloaders.bert import BertDataloader
from trainers.bert import BERTTrainer
from utils import *

def validate():
    export_root = setup_test(args)
    dataset = dataset_factory(args)
    train_loader, val_loader = BertDataloader(args, dataset).get_pretrain_dataloaders()
    model = BERTModel(args)
    trainer = BERTTrainer(args, model=model, train_loader=train_loader, val_loader=val_loader,test_loader=None, export_root=export_root)
    best_model = torch.load(os.path.join('./experiments/pretrain_2023_04_14/', 'models', 'best_acc_model.pth')).get('model_state_dict')
    model.load_state_dict(best_model)
    trainer.validate(1,1)

if __name__ == '__main__':
    if args.mode == 'train':
        validate()
    else:
        raise ValueError('Invalid mode')