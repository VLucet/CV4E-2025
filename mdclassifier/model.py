import torch.nn as nn
import torchvision.models as models


class CustomResnet101(nn.Module):

    def __init__(self, num_classes, freezed=True):
        """
        Constructor of the model. Here, we initialize the model's
        architecture (layers).
        """
        super(CustomResnet101, self).__init__()

        self.feature_extractor = models.resnet18(
            weights="DEFAULT"
        )  # use weights pre-trained on ImageNet

        if freezed:
            for param in self.parameters():
                param.requires_grad = False

        # replace the very last layer from the original, 1000-class output
        # ImageNet to a new one that outputs num_classes
        # number of input dimensions to last (classifier) layer
        in_features = self.feature_extractor.fc.in_features
        # discard last layer...
        self.feature_extractor.fc = nn.Identity()

        self.classifier = nn.Linear(in_features, num_classes)           # ...and create a new one
        # add a set of layers
        # self.classifier = nn.Sequential(
        #     nn.Linear(
        #         in_features, 1024
        #     ),  # dense layer takes a 2048-dim input and outputs 1024-dim
        #     nn.ReLU(inplace=True),  # ReLU activation introduces non-linearity
        #     nn.Dropout(0.3),  # for regularization
        #     nn.Linear(
        #         1024, num_classes
        #     ),  # final dense layer outputs x-dim corresponding to our target classes
        # )

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
