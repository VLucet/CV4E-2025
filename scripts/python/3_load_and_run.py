# Libraries
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
from tqdm import tqdm

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms

from torch import nn
import torchvision.models as models

from sklearn.metrics import ConfusionMatrixDisplay

import os

#############################################################

dat_merged = pd.read_csv("data/tabular/all_dat_merged.csv")
dat_train = pd.read_csv("data/split/dat_train.csv")
dat_val = pd.read_csv("data/split/dat_val.csv")

dat_labs_lookup = pd.read_csv("data/tabular/labels_lookup.csv")
# dat_test = pd.read_csv("data/split/dat_test.csv")

number_of_categories = dat_merged.label_group.nunique()

do_train = True
do_predict = True
frozen = True
# the_basepath = os.environ['DATASTORE']
the_basepath = "/mnt/class_data/group4/val/trailcam/"

batch_size = 32
epochs = 10
random_state = 77

model_name = "resnet101"

cuda_avail = torch.cuda.is_available()
print("Cuda available? " + str(cuda_avail))
device = torch.device("cuda" if cuda_avail else "cpu")
print("Device is : " + str(device))

run_name = model_name + "_e" + str(epochs) + "_b" + str(batch_size) + "_" + \
    ("frozen_" if frozen else "not_frozen_")
print(run_name)

#############################################################

# Subsampling
dat_train = dat_train.sample(frac=0.1)
dat_val = dat_val.sample(frac=0.1)

x_train = dat_train.crop_path
x_eval = dat_val.crop_path
y_train = dat_train.label_id
y_eval = dat_val.label_id

print(x_train.size, y_train.size, x_eval.size, y_eval.size)

#############################################################

class ImagesDataset(Dataset):
    """Reads in an image, transforms pixel values, and serves
    a dictionary containing the image id, image tensors, and label.
    """

    def __init__(self, basepath, x_df, y_df=None, device="cpu", model="resnet"):
        self.basepath = basepath
        self.data = x_df
        self.label = y_df
        
        if model in ["resnet50", "resnet101"]:
            self.transform = transforms.Compose(
                [
                    transforms.Resize((256, 256)),
                    # transforms.CenterCrop((224, 224)),
                    # transforms.Resize((384, 384)),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
                    ),
                ]
            )
        elif model == "other":
            self.transform = None
        else:
            self.transform = None
        
        self.device = device

    def __getitem__(self, index):
      
        image_path = self.data.iloc[index]
        
        if self.basepath is not None:
            image_path = os.path.join(self.basepath, image_path)
          
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image).to(self.device)
        image_id = self.data.index[index]
        
        # if we don't have labels (e.g. for test set) just return the image and image id
        if self.label is None:
            sample = {"image_id": image_id, "image": image}
        else:
            label = torch.tensor(
                self.label.iloc[index], dtype=torch.long
            ).to(self.device)
            sample = {"image_id": image_id, "image": image, "label": label}
        return sample

    def __len__(self):
        return len(self.data)

#############################################################

train_dataset = ImagesDataset(the_basepath, x_train, y_train, device=device, model=model_name)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size)

eval_dataset = ImagesDataset(the_basepath, x_eval, y_eval, device=device, model=model_name)
eval_dataloader = DataLoader(eval_dataset, batch_size=batch_size)

#############################################################

if model_name in ["resnet50", "resnet101"]:
    if model_name == "resnet50":
        model = models.resnet50(weights='DEFAULT')
    elif model_name == "resnet101":
        model = models.resnet101(weights='DEFAULT')

for param in model.parameters():
    param.requires_grad = False

if model_name in ["resnet50", "resnet101"]:
    model.fc = nn.Sequential(
        nn.Linear(2048, 1000),  # dense layer takes a 2048-dim input and outputs 100-dim
        nn.ReLU(inplace=True),  # ReLU activation introduces non-linearity
        nn.Dropout(0.5),  # common technique to mitigate overfitting
        nn.Linear(1000, 100),
        nn.ReLU(inplace=True),  # ReLU activation introduces non-linearity
        nn.Dropout(0.5),
        nn.Linear(
            100, number_of_categories
        ),  # final dense layer outputs x-dim corresponding to our target classes
    )
    
for param in model.fc.parameters():
    param.requires_grad = True

model = model.to(device)

#############################################################

import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

#############################################################

if do_train: 

    num_epochs = epochs

    tracking_loss = {}
    validating_loss = {}

    for epoch in range(1, num_epochs + 1):
        print(f"Starting epoch {epoch}")

        # iterate through the dataloader batches. tqdm keeps track of progress.
        for batch_n, batch in tqdm(
            enumerate(train_dataloader), total=len(train_dataloader)
        ):
          
            # Make sure model is in training mode
            model.train()

            # 1) zero out the parameter gradients so that gradients from previous batches are not used in this step
            optimizer.zero_grad()

            # 2) run the foward step on this batch of images
            outputs = model(batch["image"]) # ;print(outputs.shape)

            # 3) compute the loss
            loss = criterion(outputs, batch["label"])
            tracking_loss[(epoch, batch_n)] = float(loss)

            # 4) compute our gradients
            loss.backward()
            
            # 5) update our weights
            optimizer.step()
            
        # 6) Compute Validation loss
        with torch.no_grad():
          
            for batch_n2, batch2 in tqdm(
                 enumerate(eval_dataloader), total=len(eval_dataloader)
            ):
          
                # Make sure model is in evaluation mode
                model.eval()
                
                # 1) Evaluate the model
                outputs = model(batch2["image"])
            
                # 3) compute the loss
                loss = criterion(outputs, batch2["label"])
                validating_loss[(epoch, batch_n2)] = float(loss)

    tracking_loss = pd.Series(tracking_loss).groupby(level=0).mean()
    validating_loss = pd.Series(validating_loss).groupby(level=0).mean()

    plt.figure(figsize=(10, 5))
    
    tracking_loss.plot(label="tracking loss")
    # tracking_loss.rolling(center=True, min_periods=1, window=10).mean().plot(
    #     label="tracking loss (moving avg)"
    # )
    
    validating_loss.plot(label="validating loss")
    # validating_loss.rolling(center=True, min_periods=1, window=10).mean().plot(
    #     label="validating loss (moving avg)"
    # )
    
    plt.xlabel("(Epoch, Batch)")
    plt.ylabel("Loss")
    plt.legend(loc=0)
    # plt.show()
    
    plt.savefig("figures/" + run_name + "loss.png")

    print("TRAINING DONE")

    torch.save(model, "models/" + run_name + "model.pth")

    print("MODEL SAVED")

else:
  
    model = torch.load("models/" + run_name + "model.pth")
    
    print("MODEL LOADED")

#############################################################

# put the model in eval mode so we don't update any parameters
model = model.eval()

if do_predict:

    preds_collector = []

    # we aren't updating our weights so no need to calculate gradients
    with torch.no_grad():
        for batch in tqdm(eval_dataloader, total=len(eval_dataloader)):
          
            # 1) run the forward step
            logits = model.forward(batch["image"])
            
            # 2) apply softmax so that model outputs are in range [0,1]
            preds = nn.functional.softmax(logits, dim=1)

            # 3) store this batch's predictions in df
            # note that PyTorch Tensors need to first be detached from their computational graph before converting to numpy arrays
            preds_df = pd.DataFrame(
                preds.detach().cpu().numpy(),
                index=batch["image_id"].numpy(),
                # columns=species_labels
            )
            preds_collector.append(preds_df)

    eval_preds_df = pd.concat(preds_collector)

    eval_preds_df.to_csv("outputs/" + run_name + "predictions.csv")

# eval_preds_df = pd.read_csv("predictions.csv", index_col=0)

#############################################################

print(eval_preds_df.head())

eval_predictions = eval_preds_df.idxmax(axis=1)
print(eval_predictions.head())

print("Predicted vs true labels (eval):")
pred_vs_eval = eval_predictions.value_counts().to_frame().rename(
    columns = {'count': 'predicted'}).merge(y_eval.value_counts().to_frame().rename(
        columns = {'count': 'truth'}), left_index=True, right_index=True).reset_index(names=["label_id"]) \
        .merge(dat_labs_lookup, on='label_id', how = 'left')
print(pred_vs_eval)

# Overall accuracy
correct = (eval_predictions == y_eval).sum()
accuracy = correct / len(eval_predictions)
print("Overall accuracy:")
print(accuracy)

# Species metrics
print("Species F1 scores:")
for species in pred_vs_eval.label_id:
    print(species)
    
    correct = ((eval_predictions == species) == (y_eval == species)).sum()
    sp_accuracy = correct / len(eval_predictions)
    # print(sp_accuracy)
    
    P = (y_eval == species).sum()
    N = (y_eval != species).sum()
    PP = (eval_predictions == species).sum()
    PN = (eval_predictions != species).sum()
    TP = (eval_predictions == species)[y_eval == species].sum()
    FP = (eval_predictions == species)[y_eval != species].sum()
    TN = (eval_predictions != species)[y_eval != species].sum()
    FN = (eval_predictions != species)[y_eval == species].sum()
     
    recall = TP/P
    precision = TP/PP
    F1 = 2*((precision*recall)/(precision+recall))
    # F1 == (2*TP)/(2*TP + FP + FN)
    print(F1)

## This needs torchmetrics
# eval_preds_df.columns = range(0,9)
# eval_preds_idx = torch.tensor(eval_preds_df.idxmax(axis=1).values)
# y_eval.columns = range(0,9)
# y_eval_idx = torch.tensor(y_eval.idxmax(axis=1).values)
# multiclass_f1_score(eval_preds_idx, y_eval_idx, num_classes=number_of_categories)

# CM
fig, ax = plt.subplots(figsize=(10, 10))
cm = ConfusionMatrixDisplay.from_predictions(
    y_eval,
    eval_preds_df.idxmax(axis=1),
    ax=ax,
    xticks_rotation=90,
    colorbar=True,
)

fig.savefig("figures/" + run_name + "cm.png")