import torch.nn as nn
import torchvision.models as models

class CustomResnet(nn.Module):

    def __init__(self, num_classes, cfg):
        """
        Constructor of the model. Here, we initialize the model's
        architecture (layers).
        """

        super(CustomResnet, self).__init__()
        
        if cfg["model_name"] == "resnet18":
            self.feature_extractor = models.resnet18(
                weights="DEFAULT"
            )  # use weights pre-trained on ImageNet
        elif cfg["model_name"] == "resnet34":
            self.feature_extractor = models.resnet34(
                weights="DEFAULT"
            )  # use weights pre-trained on ImageNet
        else:
            raise ValueError("Resnet type not found")

        # replace the very last layer from the original, 1000-class output
        # ImageNet to a new one that outputs num_classes
        # number of input dimensions to last (classifier) layer
        in_features = self.feature_extractor.fc.in_features
        # discard last layer...
        self.feature_extractor.fc = nn.Identity()
        # ...and create a new one
        self.classifier = nn.Linear(in_features, num_classes)

        if cfg["freezed"]:
            for param in self.feature_extractor.parameters():
                param.requires_grad = False
            for param in self.classifier.parameters():
                param.requires_grad = True

    def forward(self, x):
        """
        Forward pass. Here, we define how to apply our model. It's basically
        applying our modified ResNet-18 on the input tensor ("x") and then
        apply the final classifier layer on the ResNet-18 output to get our
        num_classes prediction.
        """
        # x.size(): [B x 3 x W x H]
        features = self.feature_extractor(x)  # features.size(): [B x 512 x W x H]
        prediction = self.classifier(features)  # prediction.size(): [B x num_classes]

        return prediction

# nn.Sequential(
#     nn.Linear(
#         in_features, 1024
#     ),  # dense layer
#     nn.ReLU(),  # ReLU activation introduces non-linearity
#     nn.Dropout(cfg['dropout_rate']),  # for regularization
#     nn.Linear(
#         1024, num_classes
#     ),  # final dense layer
# )