from json import load
from pathlib import PurePath

import cv2
import numpy
from lxml.etree import _Element, _ElementTree, parse
from numpy import ndarray
from pandas import DataFrame
from progress.bar import Bar

blackWhiteThreshold: int = 128


def loadJSON(filepath: PurePath) -> dict:
    data: dict = {}

    with open(filepath, "r") as jsonFile:
        fileData: dict = load(jsonFile)
        jsonFile.close()

    topLevelKeys: list = fileData.keys()

    tlk: str
    for tlk in topLevelKeys:
        data.update(fileData[tlk])

    return data


def loadBBoxFromFile(filepath: str) -> dict | None:
    try:
        et: _ElementTree = parse(filepath)
    except OSError:
        return None

    objectElement: _Element = et.find(path="object")
    bndBoxElement: _Element = objectElement.find(path="bndbox")

    coordinates: dict = {}
    idx: int
    for idx in range(len(bndBoxElement)):
        match bndBoxElement[idx].tag:
            case "xmin":
                coordinates["xmin"] = float(bndBoxElement[idx].text)
            case "ymin":
                coordinates["ymin"] = float(bndBoxElement[idx].text)
            case "xmax":
                coordinates["xmax"] = float(bndBoxElement[idx].text)
            case "ymax":
                coordinates["ymax"] = float(bndBoxElement[idx].text)
            case _:
                print("ERROR LOADING BBOX COORDINATES:", filepath)

    return coordinates


def computeImageMetrics(imageFilePath: str, coordinates: dict) -> DataFrame:
    image: ndarray = cv2.imread(imageFilePath)
    totalImageArea: int = image.shape[0] * image.shape[1]
    totalWhitePixels: int = numpy.sum(image >= 128) / 3
    totalBlackPixels: int = numpy.sum(image < 128) / 3

    croppedImage: ndarray = image[
        round(coordinates["ymin"]) : round(coordinates["ymax"]),
        round(coordinates["xmin"]) : round(coordinates["xmax"]),
    ]
    croppedImageArea: int = croppedImage.shape[0] * croppedImage.shape[1]
    croppedWhitePixels: int = numpy.sum(croppedImage >= 128) / 3
    croppedBlackPixels: int = numpy.sum(croppedImage < 128) / 3

    diffArea: int = totalImageArea - croppedImageArea
    diffWhitePixels: int = totalWhitePixels - croppedWhitePixels
    diffBlackPixels: int = totalBlackPixels - croppedBlackPixels


def main() -> None:
    annotationsMap: PurePath = PurePath("annotationMap.json")
    data: dict = loadJSON(filepath=annotationsMap)

    with Bar(
        "Checking bounding boxes for each type of aoi map...", max=len(data.keys())
    ) as bar:
        xmlFilePath: str
        for xmlFilePath in data.keys():
            coordinates: dict | None = loadBBoxFromFile(filepath=xmlFilePath)
            if coordinates is None:
                bar.next()
                continue

            computeImageMetrics(data[xmlFilePath], coordinates)

            topLeft: tuple = (
                round(coordinates["xmin"]),
                round(coordinates["ymin"]),
            )
            bottomRight: tuple = (
                round(coordinates["xmax"]),
                round(coordinates["ymax"]),
            )

            # # image: ndarray = array(Image.open(imageFilePath).convert("RGB"))
            # # cv2.rectangle(
            # #     image, topLeft, bottomRight, color=(255, 0, 0), thickness=1
            # # )
            # cv2.imshow(xmlFilePath, image)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            quit()

        bar.next()


if __name__ == "__main__":
    main()
