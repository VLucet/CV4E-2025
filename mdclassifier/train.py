# MAIN SCRIPT

import sys
sys.path.append(".")

from skbio import diversity as sd
from torch.utils.data import DataLoader
from dataset import MDclassDataset
from model import CustomResnet
from torch import nn
from util import init_seed
from tqdm import trange
from myutils.MDSplit import *

import torch
import os
# import yaml
import argparse
import wandb
import torchmetrics

import torch.optim as optim
import pandas as pd


#############################################################


def split_test_train_val(all_dat, species_group_ord, split_name, write=True):

    all_dat_summ = (
        all_dat.groupby(by=["label_group", "loc_id"], as_index=False, sort=False)
        .size()
        .sort_values(["label_group", "size"], ascending=[True, False])
    )

    # Make the column categorical for ordering
    all_dat_summ["label_group"] = pd.Categorical(
        all_dat_summ["label_group"], categories=species_group_ord["label_group"]
    )

    # Count to get best loc
    dat_pivot = (
        all_dat_summ.pivot_table(
            index="loc_id", columns="label_group", values="size", aggfunc="sum"
        )
        .reset_index()
        .fillna(0)
    )
    dat_locs = dat_pivot["loc_id"]
    counts_table = dat_pivot.drop(["loc_id"], axis=1)
    counts_table = counts_table.rename_axis(None, axis=1)

    # Compute shannon div
    shannon_div = sd.alpha_diversity("shannon", counts_table)
    best_loc = dat_locs[shannon_div.argmax()]

    # Filter out testloc
    dat_test = all_dat.query(f'loc_id == "{best_loc}"')
    dat_train_val = all_dat.query(f'loc_id != "{best_loc}"')

    if write:
        dat_test.to_csv(f"data/tabular/splits/{split_name}/dat_test.csv", index=False)
        dat_train_val.to_csv(
            f"data/tabular/splits/{split_name}/dat_train_val.csv", index=False
        )

    return dat_test, dat_train_val


def split_data(all_dat, species_group_ord, cfg, split_name, write=True):

    print(f'Splitting data following name "{split_name}"')

    if cfg["data_frac"] != 1.0:
        print("Subsampling data")
        all_dat = all_dat.sample(frac=cfg["data_frac"])

    if os.path.exists(f"data/tabular/splits/{split_name}"):

        dat_train = pd.read_csv(f"data/tabular/splits/{split_name}/dat_train.csv")
        dat_val = pd.read_csv(f"data/tabular/splits/{split_name}/dat_val.csv")
        dat_test = pd.read_csv(f"data/tabular/splits/{split_name}/dat_test.csv")

    else:

        os.makedirs(f"data/tabular/splits/{split_name}", exist_ok=True)

        dat_test, dat_train_val = split_test_train_val(
            all_dat, species_group_ord, split_name
        )

        # Run the split
        dat_tt_tab_dict = (
            dat_train_val.groupby(
                by=["label_group", "loc_id"], as_index=False, sort=False
            )
            .size()
            .sort_values(["label_group", "size"], ascending=[True, False])
            .pivot_table(
                index="label_group", columns="loc_id", values="size", aggfunc="sum"
            )
            .reset_index()
            .set_index("label_group")
            .rename_axis(None, axis=1)
            .fillna(0)
        ).to_dict()

        the_split = split_locations_into_train_val(
            dat_tt_tab_dict,
            n_random_seeds=10000,
            target_val_fraction=(cfg["train_val_split"]),
            category_to_max_allowable_error=None,
            category_to_error_weight=None,
            default_max_allowable_error=0.5,
        )

        with open(f"data/tabular/splits/{split_name}/split.txt", "w") as f:
            print(the_split, file=f)

        dat_train = dat_train_val.query(f"loc_id not in {the_split[0]}")
        dat_val = dat_train_val.query(f"loc_id in {the_split[0]}")

        if write:
            dat_train.to_csv(
                f"data/tabular/splits/{split_name}/dat_train.csv", index=False
            )
            dat_val.to_csv(f"data/tabular/splits/{split_name}/dat_val.csv", index=False)

    return dat_train, dat_val, dat_test


def create_dataloader(cfg, x_df, y_df, shuffle=True):
    """
    Loads a dataset according to the provided split and wraps it in a
    PyTorch DataLoader object.
    """
    # create an object instance of our CTDataset class
    dataset_instance = MDclassDataset(cfg, x_df, y_df)

    dataLoader = DataLoader(
        dataset=dataset_instance,
        batch_size=cfg["batch_size"],
        num_workers=cfg["num_workers"],
        shuffle=shuffle,
    )

    return dataLoader


def load_model(cfg, number_of_categories):
    """
    Creates a model instance and loads the latest model state weights.
    """
    if cfg["model_name"] in ["resnet18", "resnet101"]:
        model_instance = CustomResnet(number_of_categories, cfg)
    else:
        raise "Model class not found"

    # no save state found; start anew
    print("Starting new model")
    start_epoch = 0

    return model_instance, start_epoch


# def save_model(cfg, epoch, model, stats, run_name, last=False):
#     # make sure save directory exists; create if not
#     os.makedirs(f"runs/{run_name}", exist_ok=True)

#     # get model parameters and add to stats...
#     stats["model"] = model.state_dict()
#     stats["epoch"] = epoch

#     # ...and save
#     if last:
#         torch.save(stats, open(f"runs/{run_name}/{epoch}.pt", "wb"))
#     else:
#         torch.save(stats, open(f"runs/{run_name}/best.pt", "wb"))
    
#     # also save config file if not present
#     cfpath = f"runs/{run_name}/{run_name}_config.yaml"
#     if not os.path.exists(cfpath):
#         with open(cfpath, "w") as f:
#             yaml.dump(cfg, f)


def setup_optimizer(cfg, model):
    """
    The optimizer is what applies the gradients to the parameters and makes
    the model learn on the dataset.
    """

    if cfg["optimizer"] == "sgd":
        optimizer = optim.SGD(
            model.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"]
        )
    elif cfg["optimizer"] == "adam":
        optimizer = optim.Adam(
                model.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"]
            )
    elif cfg["optimizer"] == "adamw":
        optimizer = optim.AdamW(
                model.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"]
            )
    else:
        raise "Optimizer not found"
    
    return optimizer


def train(cfg, dataLoader, model, optimizer, number_of_categories):
    """
    Our actual training function.
    """

    device = cfg["device"]

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

    # Preds and labels     
    preds_total = []
    labels_total = []

    # iterate over dataLoader
    # see the last line of file "dataset.py" where we return the image tensor (data) and label
    for batch_n, batch in enumerate(dataLoader):

        # put data and labels on device
        data, labels = batch["image"].to(device), batch["label"].to(device)

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

        # Accumulate
        preds_total.append(pred_label)
        labels_total.append(labels)

        progressBar.set_description(
            "[Train] Loss: {:.2f}; OA: {:.2f}%".format(
                loss_total / (batch_n + 1), 100 * oa_total / (batch_n + 1)
            )
        )
        progressBar.update(1)

    # end of epoch; finalize
    progressBar.close()

    # end of epoch; finalize
    # shorthand notation for: loss_total = loss_total / len(dataLoader)
    loss_total /= len(dataLoader)
    oa_total /= len(dataLoader)

    # metrics calculations
    preds_total = torch.concat(preds_total)
    labels_total = torch.concat(labels_total)

    dict_metrics, cfm_fig = compute_metrics(preds_total, labels_total, number_of_categories, device)

    return loss_total, oa_total, dict_metrics, cfm_fig


def validate(cfg, dataLoader, model, number_of_categories):
    """
    Validation function. Note that this looks almost the same as the training
    function, except that we don't use any optimizer or gradient steps.
    """

    device = cfg["device"]
    model.to(device)

    # put the model into evaluation mode
    model.eval()

    # we still need a criterion to calculate the validation loss
    criterion = nn.CrossEntropyLoss()

    # running averages
    # for now, we just log the loss and overall accuracy (OA)
    loss_total, oa_total = 0.0, 0.0

    # iterate over dataLoader
    progressBar = trange(len(dataLoader))

    # Preds and labels     
    preds_total = []
    labels_total = []

    # don't calculate intermediate gradient steps: we don't need them, so this saves memory and is faster
    with torch.no_grad():
        for batch_n, batch in enumerate(dataLoader):

            # put data and labels on device
            data, labels = batch["image"].to(device), batch["label"].to(device)

            # forward pass
            prediction = model(data)

            # loss
            loss = criterion(prediction, labels)

            # log statistics
            loss_total += loss.item()

            pred_label = torch.argmax(prediction, dim=1)
            oa = torch.mean((pred_label == labels).float())
            oa_total += oa.item()

            # Accumulate
            preds_total.append(pred_label)
            labels_total.append(labels)

            progressBar.set_description(
                "[Val] Loss: {:.2f}; OA: {:.2f}%".format(
                    loss_total / (batch_n + 1), 100 * oa_total / (batch_n + 1)
                )
            )
            progressBar.update(1)

        progressBar.close()

    # end of epoch; finalize
    loss_total /= len(dataLoader)
    oa_total /= len(dataLoader)

    # metrics calculations
    preds_total = torch.concat(preds_total)
    labels_total = torch.concat(labels_total)

    dict_metrics, cfm_fig = compute_metrics(preds_total, labels_total, number_of_categories, device)

    return loss_total, oa_total, dict_metrics, cfm_fig


def compute_metrics(preds_total, labels_total, number_of_categories, device):

    # Species specific
    acc_tm = torchmetrics.classification.Accuracy(task="multiclass", 
                                                  num_classes=number_of_categories).to(device)
    rec_tm = torchmetrics.classification.Recall(task="multiclass", 
                                                num_classes=number_of_categories).to(device)
    pre_tm = torchmetrics.classification.Precision(task="multiclass", 
                                                   num_classes=number_of_categories).to(device)
    dict_metrics = {}
    for cls in range(number_of_categories):
        preds_bin = torch.eq(preds_total, cls)
        labels_bin = torch.eq(labels_total, cls)
        dict_metrics[cls] = { 
            "tm_acc": acc_tm(preds_bin, labels_bin),
            "rec_tm": rec_tm(preds_bin, labels_bin),
            "pre_tm": pre_tm(preds_bin, labels_bin),
        }

    # Confusion matrix
    cfm = torchmetrics.classification.MulticlassConfusionMatrix(
        num_classes=number_of_categories, normalize="true").to(device)
    cfm.update(preds_total, labels_total)
    cfm_fig = cfm.plot()[0]
    cfm_fig.set_size_inches(8,8)

    return dict_metrics, cfm_fig


#############################################################


def make_split_name(cfg):
    return "frac" + str(cfg["data_frac"]) + "_" + "split" + str(cfg["train_val_split"])


# def make_run_name(model_name, cfg):
#     run_name = (
#         model_name
#         + "_"
#         + str(cfg["num_epochs"])
#         + "e"
#         + "_"
#         + str(cfg["batch_size"])
#         + "bs"
#         + "_"
#         + str(cfg["learning_rate"])
#         + "lr"
#         + "_"
#         + str(cfg["weight_decay"])
#         + "wd"
#     )
#     return run_name


#############################################################


def main(cfg):

    # Argument parser for command-line arguments:
    # python ct_classifier/train.py --config configs/exp_resnet18.yaml
    # parser = argparse.ArgumentParser(description="Train deep learning model.")
    # parser.add_argument("--config", help="Path to config file")
    # args = parser.parse_args()

    # load config
    # print(f'Using config "{args.config}"')
    # cfg = yaml.safe_load(open(args.config, "r"))

    # init random number generator seed (set at the start)
    init_seed(cfg["seed"])

    # check if GPU is available
    device = cfg["device"]
    if device != "cpu" and not torch.cuda.is_available():
        print(
            f'WARNING: device set to "{device}" but CUDA not available; falling back to CPU...'
        )
        cfg["device"] = "cpu"

    # Load data
    dat_merged = pd.read_csv("data/tabular/all_dat_merged.csv")
    species_group_ord = pd.read_csv("data/tabular/species_groups_ord.csv")
    dat_labs_lookup = pd.read_csv("data/tabular/labels_lookup.csv")\
        .drop("size", axis = 1) \
        .set_index("label_id") \
        .to_dict()['label_group']
    
    # Split data
    split_name = make_split_name(cfg)
    dat_train, dat_val, dat_test = split_data(dat_merged, species_group_ord, cfg, split_name)

    x_train = dat_train.crop_path
    x_eval = dat_val.crop_path
    x_test = dat_test.crop_path
    y_train = dat_train.label_id
    y_eval = dat_val.label_id
    y_test = dat_test.label_id

    number_of_categories = len(dat_labs_lookup)

    # Make run_name
    # model_name = cfg["model_name"]
    # run_name = make_run_name(model_name, cfg) + "_" + split_name

    # start wandb
    # start a new wandb run to track this script
    wandb.init(
        # set the wandb project where this run will be logged
        project=cfg["wandb_project"],
        # track hyperparameters and run metadata
        config=cfg,
        # name
        # name=run_name,
    )

    # initialize data loaders for training and validation set
    dl_train = create_dataloader(cfg, x_train, y_train)
    dl_val = create_dataloader(cfg, x_eval, y_eval, shuffle=False)
    dl_test = create_dataloader(cfg, x_test, y_test, shuffle=False)

    # initialize model
    model, current_epoch = load_model(cfg, number_of_categories)

    # set up model optimizer
    optim = setup_optimizer(cfg, model)

    # Tracking of loss_val
    # stop_track = 1000

    # we have everything now: data loaders, model, optimizer; let's do the epochs!
    numEpochs = cfg["num_epochs"]
    while current_epoch < numEpochs:
        current_epoch += 1
        print(f"Epoch {current_epoch}/{numEpochs}")

        loss_train, oa_train, dict_metrics_train, cfm_train = train(cfg, dl_train, model, 
                                                                    optim, number_of_categories)
        loss_val, oa_val, dict_metrics_val, cfm_val  = validate(cfg, dl_val, model, 
                                                                number_of_categories)
        loss_test, oa_test, dict_metrics_test, cfm_test  = validate(cfg, dl_test, model, 
                                                                    number_of_categories)
        
        # log metrics to wandb
        wandb.log({
                "loss_train": loss_train,
                "oa_train": oa_train,
                "loss_val": loss_val,
                "oa_val": oa_val,
                "loss_test": loss_test,
                "oa_test": oa_test,
                "train": {dat_labs_lookup[key]: value for key, value in dict_metrics_train.items()},
                "val": {dat_labs_lookup[key]: value for key, value in dict_metrics_val.items()},
                "test": {dat_labs_lookup[key]: value for key, value in dict_metrics_test.items()},
                "cfm_train": wandb.Image(cfm_train),
                "cfm_val": wandb.Image(cfm_val),
                "cfm_test": wandb.Image(cfm_test)
            })

        # combine stats and save

        # if loss_val < stop_track:
        #     stop_track = loss_val
        #     save_model(cfg, current_epoch, model, stats, run_name)
        # if current_epoch == numEpochs:
        #     save_model(cfg, current_epoch, model, stats, run_name)

    # [optional] finish the wandb run, necessary in notebooks
    wandb.finish()


def parse_args():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--wandb_project', type=str)
    parser.add_argument('--basepath', type=str)
    parser.add_argument('--train_val_split', type=float)
    parser.add_argument('--data_frac', type=float)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--device', type=str)
    parser.add_argument('--num_workers', type=int)
    parser.add_argument('--model_name', type=str)
    parser.add_argument('--freezed', type=bool)
    parser.add_argument('--image_size', type=int)
    parser.add_argument('--num_epochs', type=int)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--learning_rate', type=float)
    parser.add_argument('--weight_decay', type=float)
    parser.add_argument('--optimizer', type=str)
    parser.add_argument('--rotation', type=int)
    parser.add_argument('--horizontal_flip', type=float)

    return vars(parser.parse_args())


#############################################################


if __name__ == "__main__":
    # This block only gets executed if you call the "train.py" script directly
    # (i.e., "python ct_classifier/train.py").
    cfg = parse_args()
    main(cfg)
