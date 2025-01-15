# MAIN SCRIPT

from torch.utils.data import DataLoader
from dataset import MDclassDataset
from model import CustomResnet101
from torch import nn
from util import init_seed
from tqdm import trange

import glob
import torch
import os
import yaml
import argparse
import wandb

import torch.optim as optim
import pandas as pd

def create_dataloader(cfg, x_df, y_df, split, model):
    '''
        Loads a dataset according to the provided split and wraps it in a
        PyTorch DataLoader object.
    '''
    # create an object instance of our CTDataset class
    dataset_instance = MDclassDataset(cfg, x_df, y_df, split, model)

    dataLoader = DataLoader(
            dataset=dataset_instance,
            batch_size=cfg['batch_size'],
            num_workers=cfg['num_workers'],
            shuffle=True
        )
    
    return dataLoader


def load_model(cfg, number_of_categories):
    '''
        Creates a model instance and loads the latest model state weights.
    '''
    model_instance = CustomResnet101(number_of_categories)  

    # load latest model state
    model_states = glob.glob('model_states/*.pt')
    
    if len(model_states):
        # at least one save state found; get latest
        model_epochs = [int(m.replace('model_states/','').replace('.pt','')) for m in model_states]
        start_epoch = max(model_epochs)

        # load state dict and apply weights to model
        print(f'Resuming from epoch {start_epoch}')
        state = torch.load(open(f'model_states/{start_epoch}.pt', 'rb'), map_location='cpu')
        model_instance.load_state_dict(state['model'])

    else:
        # no save state found; start anew
        print('Starting new model')
        start_epoch = 0

    return model_instance, start_epoch


def save_model(cfg, epoch, model, stats):
    # make sure save directory exists; create if not
    os.makedirs('model_states', exist_ok=True)

    # get model parameters and add to stats...
    stats['model'] = model.state_dict()

    # ...and save
    torch.save(stats, open(f'model_states/{epoch}.pt', 'wb'))
    
    # also save config file if not present
    cfpath = 'model_states/config.yaml'
    if not os.path.exists(cfpath):
        with open(cfpath, 'w') as f:
            yaml.dump(cfg, f)


def setup_optimizer(cfg, model):
    '''
        The optimizer is what applies the gradients to the parameters and makes
        the model learn on the dataset.
    '''
    optimizer = optim.Adam(model.parameters(),
                    lr=cfg['learning_rate'],
                    weight_decay=cfg['weight_decay'])
    return optimizer


def train(cfg, dataLoader, model, optimizer):
    '''
        Our actual training function.
    '''
    
    device = cfg['device']

    # put model on device
    model.to(device)
    
    # put the model into training mode
    # this is required for some layers that behave differently during training
    # and validation (examples: Batch Normalization, Dropout, etc.)
    model.train()

    # loss function
    #  note: if you're doing multi target classification, use nn.BCEWithLogitsLoss() and convert labels to float
    criterion = nn.CrossEntropyLoss()

    # running averages
    # for now, we just log the loss and overall accuracy (OA)
    loss_total, oa_total = 0.0, 0.0
        
    # iterate over dataLoader
    progressBar = trange(len(dataLoader))

    # iterate over dataLoader
    # see the last line of file "dataset.py" where we return the image tensor (data) and label
    for batch_n, batch in enumerate(dataLoader):

        # put data and labels on device
        data, labels =  batch["image"].to(device), batch["label"].to(device)

        # forward pass
        prediction = model(data)

        # reset gradients to zero
        optimizer.zero_grad()

        # loss
        loss = criterion(prediction, labels)

        # backward pass (calculate gradients of current batch)
        loss.backward()

        # apply gradients to model parameters
        optimizer.step()

        # log statistics
        # the .item() command retrieves the value of a single-valued tensor, regardless of its data type and device of tensor
        loss_total += loss.item()                    

        # the predicted label is the one at position (class index) with highest predicted value
        pred_label = torch.argmax(prediction, dim=1)
        # OA: number of correct predictions divided by batch size (i.e., average/mean)
        oa = torch.mean((pred_label == labels).float())
        oa_total += oa.item()

        progressBar.set_description(
            '[Train] Loss: {:.2f}; OA: {:.2f}%'.format(
                loss_total/(batch_n+1),
                100*oa_total/(batch_n+1)
            )
        )
        progressBar.update(1)

    # end of epoch; finalize
    progressBar.close()
    
    # end of epoch; finalize
    # shorthand notation for: loss_total = loss_total / len(dataLoader)
    loss_total /= len(dataLoader)
    oa_total /= len(dataLoader)

    return loss_total, oa_total


def validate(cfg, dataLoader, model):
    '''
        Validation function. Note that this looks almost the same as the training
        function, except that we don't use any optimizer or gradient steps.
    '''
    
    device = cfg['device']
    model.to(device)

    # put the model into evaluation mode
    # see lines 103-106 above
    model.eval()
    
    # we still need a criterion to calculate the validation loss
    criterion = nn.CrossEntropyLoss()   

    # running averages
    # for now, we just log the loss and overall accuracy (OA)
    loss_total, oa_total = 0.0, 0.0
    
    # iterate over dataLoader
    progressBar = trange(len(dataLoader))
    
    # don't calculate intermediate gradient steps: we don't need them, so this saves memory and is faster
    with torch.no_grad():            
        for batch_n, batch in enumerate(dataLoader):

            # put data and labels on device
            data, labels =  batch["image"].to(device), batch["label"].to(device)

            # forward pass
            prediction = model(data)

            # loss
            loss = criterion(prediction, labels)

            # log statistics
            loss_total += loss.item()

            pred_label = torch.argmax(prediction, dim=1)
            oa = torch.mean((pred_label == labels).float())
            oa_total += oa.item()

            progressBar.set_description(
                '[Val] Loss: {:.2f}; OA: {:.2f}%'.format(
                    loss_total/(batch_n+1),
                    100*oa_total/(batch_n+1)
                )
            )
            progressBar.update(1)

        progressBar.close()
    
    # end of epoch; finalize
    loss_total /= len(dataLoader)
    oa_total /= len(dataLoader)

    return loss_total, oa_total

#############################################################

def main():

    # Argument parser for command-line arguments:
    # python ct_classifier/train.py --config configs/exp_resnet18.yaml
    parser = argparse.ArgumentParser(description='Train deep learning model.')
    parser.add_argument('--config', help='Path to config file')
    args = parser.parse_args()

    # load config
    print(f'Using config "{args.config}"')
    cfg = yaml.safe_load(open(args.config, 'r'))

    # init random number generator seed (set at the start)
    init_seed(cfg.get('seed', None))

    # check if GPU is available
    device = cfg['device']
    if device != 'cpu' and not torch.cuda.is_available():
        print(f'WARNING: device set to "{device}" but CUDA not available; falling back to CPU...')
        cfg['device'] = 'cpu'

    # Load data
    dat_merged = pd.read_csv("data/tabular/all_dat_merged.csv")
    dat_train = pd.read_csv("data/split/dat_train.csv")
    dat_val = pd.read_csv("data/split/dat_val.csv")
    # dat_test = pd.read_csv("data/split/dat_test.csv")
    # dat_labs_lookup = pd.read_csv("data/tabular/labels_lookup.csv")
    
    number_of_categories = dat_merged.label_group.nunique()
    model_name = "resnet101"
    
    dat_train = dat_train.sample(frac=0.1)
    dat_val = dat_val.sample(frac=0.1)

    x_train = dat_train.crop_path
    x_eval = dat_val.crop_path
    y_train = dat_train.label_id
    y_eval = dat_val.label_id
    
    # start wandb
    # start a new wandb run to track this script
    wandb.init(
        # set the wandb project where this run will be logged
        project="cv4e-test",
        # track hyperparameters and run metadata
        config=cfg
    )

    # initialize data loaders for training and validation set
    dl_train = create_dataloader(cfg, x_train, y_train, split='train', 
                                 model=model_name)
    dl_val = create_dataloader(cfg, x_eval, y_eval, split='val', 
                               model=model_name)

    # initialize model
    model, current_epoch = load_model(cfg, number_of_categories)

    # set up model optimizer
    optim = setup_optimizer(cfg, model)

    # we have everything now: data loaders, model, optimizer; let's do the epochs!
    numEpochs = cfg['num_epochs']
    while current_epoch < numEpochs:
        current_epoch += 1
        print(f'Epoch {current_epoch}/{numEpochs}')

        loss_train, oa_train = train(cfg, dl_train, model, optim)
        loss_val, oa_val = validate(cfg, dl_val, model)
        
                # log metrics to wandb
        wandb.log({"loss_train": loss_train, "oa_train": oa_train,
                   "loss_val": loss_val, "oa_val": oa_val})

        # combine stats and save
        stats = {
            'loss_train': loss_train,
            'loss_val': loss_val,
            'oa_train': oa_train,
            'oa_val': oa_val
        }
        save_model(cfg, current_epoch, model, stats)
    
    # [optional] finish the wandb run, necessary in notebooks
    wandb.finish()

#############################################################        

if __name__ == '__main__':
    # This block only gets executed if you call the "train.py" script directly
    # (i.e., "python ct_classifier/train.py").
    main()