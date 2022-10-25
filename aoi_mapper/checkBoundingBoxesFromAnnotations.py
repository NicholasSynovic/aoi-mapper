from json import load

import cv2
from lxml.etree import _Element, _ElementTree, parse
from numpy import ndarray
from progress.bar import Bar
import numpy
from pathlib import PurePath

blackWhiteThreshold: int = 128

def loadJSON(filepath: PurePath)    ->  dict:
    data: dict = {}

    with open(filepath, "r") as jsonFile:
        fileData: dict = load(jsonFile)
        jsonFile.close()

    topLevelKeys: list = fileData.keys()

    tlk: str
    for tlk in topLevelKeys:
        data.update(fileData[tlk])

    return data

def loadBBoxFromFile(filepath: str)    ->  dict | None:
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

def main() -> None:
    annotationsMap: PurePath = PurePath("annotationMap.json")
    data: dict = loadJSON(filepath=annotationsMap)

    with Bar("Checking bounding boxes for each type of aoi map...", max=len(data.keys())) as bar:
        xmlFilePath: str
        for xmlFilePath in data.keys():
            coordinates: dict | None = loadBBoxFromFile(filepath=xmlFilePath)

            if coordinates is None:
                bar.next()
                continue

            topLeft: tuple = (
                round(coordinates["xmin"]),
                round(coordinates["ymin"]),
            )
            bottomRight: tuple = (
                round(coordinates["xmax"]),
                round(coordinates["ymax"]),
            )

            imageFilePath: str = data[xmlFilePath]

            image: ndarray = cv2.imread(imageFilePath)

            imageArea: int = image.shape[0] * image.shape[1]
            whitePixels: int = numpy.sum(image >= 128) / 3
            blackPixels: int = numpy.sum(image < 128) / 3
            pixelSum = blackPixels + whitePixels
            error = pixelSum - imageArea

            otherPixels: int = numpy.sum((image != 0) & (image != 255))

            print()
            print(imageFilePath)
            print("black pixels:", blackPixels)
            print("white pixels:",whitePixels)
            print()
            print("pixel sum:",pixelSum)
            print("area:", imageArea)
            print("actual:", 500 * 343)
            print("Error:", error)
            print("other pixels:", otherPixels)

            imageUniquePixelValues = numpy.unique(image)
            print(imageUniquePixelValues)

            # hist = numpy.histogram(temp)
            # print(hist)



            # wrongPixels: ndarray = numpy.where((image != 0) & (image != 255))
            # print(wrongPixels)
            # # shape: tuple = image.shape

            # totalBlackPixels: int = numpy.sum(image == 255) / 3
            # totalWhitePixels: int = numpy.sum(image == 0) / 3

            # totalPixels: int = shape[0] * shape[1]

            # print()
            # print("total bp:", totalBlackPixels)
            # print("total wp:", totalWhitePixels)
            # print("bp + wp:", totalBlackPixels + totalWhitePixels)
            # print("total p:", totalPixels)

            # print("error:", totalBlackPixels + totalWhitePixels - totalPixels)


            # image = image[
            #     round(coordinates["ymin"]) : round(coordinates["ymax"]),
            #     round(coordinates["xmin"]) : round(coordinates["xmax"]),
            # ]

            # totalBlackPixels: int = numpy.sum(image == 255) / 3
            # totalWhitePixels: int = numpy.sum(image == 0) / 3
            # totalPixels: int = (totalBlackPixels + totalWhitePixels) / 3

            # print(
            #     "\n",
            #     totalBlackPixels,
            #     totalWhitePixels,
            #     totalPixels,
            # )

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
