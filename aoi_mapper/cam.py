# Simple implementation of CAM in PyTorch for the networks such as ResNet, DenseNet, SqueezeNet, Inception

import json
from os import listdir
from os.path import join
from pathlib import PurePath
from typing import Tuple

import cv2
import numpy as np
from numpy import ndarray
from PIL import Image
from progress.bar import Bar
from torch import Tensor
from torch.autograd import Variable
from torch.nn import functional as F
from torchvision import models, transforms
from torchvision.models import (DenseNet, ResNet, ResNet18_Weights, SqueezeNet,
                                SqueezeNet1_1_Weights)

features_blobs: list = []


def loadImageNetClasses(filepath: str) -> dict:
    with open(filepath, "r") as jsonFile:
        imageNetClasses = json.load(jsonFile)
        jsonFile.close()
    return imageNetClasses


def readDirectory(dir: PurePath) -> list:
    files: list = listdir(dir)
    filepaths: list = [join(dir, f) for f in files]
    return filepaths


def loadPTM(id: int = 1) -> Tuple[SqueezeNet | ResNet | DenseNet | None, str, str]:
    finalconv_name: str
    modelName: str
    if id == 1:
        net: SqueezeNet = models.squeezenet1_1(weights=SqueezeNet1_1_Weights.DEFAULT)
        finalconv_name = "features"
        modelName = "squeezenet"
    elif id == 2:
        net: ResNet = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        finalconv_name = "layer4"
        modelName = "resnet"
    elif id == 3:
        net: DenseNet = models.densenet161(pretrained=True)
        finalconv_name = "features"
        modelName = "densenet"
    else:
        net: None = None
        finalconv_name = None
        name = None
    return (net, finalconv_name, modelName)


def hook_feature(module, input, output) -> None:
    features_blobs.append(output.data.cpu().numpy())


def returnCAM(feature_conv, weight_softmax, class_idx) -> list:
    size_upsample: tuple = (256, 256)
    bz, nc, h, w = feature_conv.shape
    output_cam: list = []
    for idx in class_idx:
        cam = weight_softmax[idx].dot(feature_conv.reshape((nc, h * w)))
        cam = cam.reshape(h, w)
        cam = cam - np.min(cam)
        cam_img = cam / np.max(cam)
        cam_img = np.uint8(255 * cam_img)
        output_cam.append(cv2.resize(cam_img, size_upsample))
    return output_cam


def main() -> None:
    modelID: int = 1
    modelCount: int = 3

    imageNetClasses: dict = loadImageNetClasses(filepath="imagenet-simple-labels.json")

    while modelID < modelCount + 1:
        net, finalconv_name, modelName = loadPTM(id=modelID)
        if net is None:
            print("Invalid ID. ID should be between 1 - 3 inclusive.")
            quit()

        net.eval()
        net._modules.get(finalconv_name).register_forward_hook(hook_feature)
        params: list = list(net.parameters())
        weight_softmax: ndarray = np.squeeze(params[-2].data.numpy())

        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
        preprocess = transforms.Compose(
            [transforms.Resize((224, 224)), transforms.ToTensor(), normalize]
        )

        imageFiles: list = []
        for foo in readDirectory(dir=PurePath("images")):
            imageFiles.append(foo)

        with Bar(
            f"Creating CAMs of images with {modelName}...", max=len(imageFiles)
        ) as bar:
            imageFile: str
            for imageFile in imageFiles:
                filename: str = PurePath(imageFile).with_suffix("").name
                outputFilename: str = filename + f"_{modelName}_cam.jpg"
                outputFilePath: str = join("cam", outputFilename)

                img_pil: Image = Image.open(imageFile)
                img_tensor = preprocess(img_pil)
                img_variable: Variable = Variable(img_tensor.unsqueeze(0))
                logit = net(img_variable)

                h_x: Tensor = F.softmax(logit, dim=1).data.squeeze()
                probs, idx = h_x.sort(0, True)
                probs: ndarray = probs.numpy()
                idx: ndarray = idx.numpy()

                # output the prediction
                # for i in range(0, 5):
                #     print("{:.3f} -> {}".format(probs[i], imageNetClasses[idx[i]]))

                # generate class activation mapping for the top1 prediction
                CAMs: list = returnCAM(features_blobs[0], weight_softmax, [idx[0]])

                # render the CAM and output
                # print("output CAM.jpg for the top1 prediction: %s" % imageNetClasses[idx[0]])
                img = cv2.imread(imageFile)
                height, width, _ = img.shape
                heatmap = cv2.applyColorMap(
                    cv2.resize(CAMs[0], (width, height)), cv2.COLORMAP_JET
                )
                result = heatmap * 0.3 + img * 0.5
                cv2.imwrite(outputFilePath, result)

                bar.next()

        modelID += 1


if __name__ == "__main__":
    main()
