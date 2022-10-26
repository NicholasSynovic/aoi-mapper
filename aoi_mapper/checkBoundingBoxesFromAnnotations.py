from json import load
from os.path import join
from pathlib import PurePath

import cv2
import numpy
import pandas
from lxml.etree import _Element, _ElementTree, parse
from numpy import array, ndarray
from pandas import DataFrame
from PIL import Image
from progress.bar import Bar

blackWhiteThreshold: int = 128


def loadJSON(filepath: PurePath) -> dict:
    with open(filepath, "r") as jsonFile:
        data: dict = load(jsonFile)
        jsonFile.close()

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
    data: dict = {}

    image: ndarray = cv2.imread(imageFilePath)
    data["totalImageArea"] = image.shape[0] * image.shape[1]
    data["totalWhitePixels"] = numpy.sum(image >= blackWhiteThreshold) / 3
    data["totalBlackPixels"] = numpy.sum(image < blackWhiteThreshold) / 3

    bboxImage: ndarray = image[
        round(coordinates["ymin"]) : round(coordinates["ymax"]),
        round(coordinates["xmin"]) : round(coordinates["xmax"]),
    ]
    data["bboxImageArea"] = bboxImage.shape[0] * bboxImage.shape[1]
    data["bboxWhitePixels"] = numpy.sum(bboxImage >= blackWhiteThreshold) / 3
    data["bboxBlackPixels"] = numpy.sum(bboxImage < blackWhiteThreshold) / 3

    data["diffArea"] = data["totalImageArea"] - data["bboxImageArea"]
    data["diffWhitePixels"] = data["totalWhitePixels"] - data["bboxWhitePixels"]
    data["diffBlackPixels"] = data["totalBlackPixels"] - data["bboxBlackPixels"]

    data["bboxWhitePixelPercentage"] = data["bboxWhitePixels"] / data["bboxImageArea"]
    data["bboxBlackPixelPercentage"] = data["bboxBlackPixels"] / data["bboxImageArea"]
    data["diffWhitePixelPercentage"]: float = (
        data["diffWhitePixels"] / data["totalImageArea"]
    )
    data["diffBlackPixelPercentage"]: float = (
        data["diffBlackPixels"] / data["totalImageArea"]
    )

    return DataFrame(data, index=[0])


def createAOIWithBBox(imageFilename: str, coordinates: dict) -> None:
    imageFilepath: PurePath = PurePath(imageFilename)

    directory: str = imageFilepath.parent
    filename: str = imageFilepath.with_suffix("").name + "_bbox.jpg"

    outFilepath: str = join(directory, filename)

    topLeft: tuple = (
        round(coordinates["xmin"]),
        round(coordinates["ymin"]),
    )
    bottomRight: tuple = (
        round(coordinates["xmax"]),
        round(coordinates["ymax"]),
    )

    image: ndarray = array(Image.open(imageFilename).convert("RGB"))
    cv2.rectangle(image, topLeft, bottomRight, color=(255, 0, 0), thickness=1)
    cv2.imwrite(filename=outFilepath, img=image)


def main() -> None:
    df: DataFrame = DataFrame()
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

            imageFile: str
            for imageFile in data[xmlFilePath]:
                df = pandas.concat([df, computeImageMetrics(imageFile, coordinates)])
                df.reset_index(drop=True, inplace=True)

                createAOIWithBBox(imageFile, coordinates)

            bar.next()
        df.T.to_json(
            "bboxStatistics.zip", double_precision=3, compression={"method": "zip"}
        )


if __name__ == "__main__":
    main()
