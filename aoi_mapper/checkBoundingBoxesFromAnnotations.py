from collections import namedtuple
from json import load

import cv2
from lxml.etree import _Element, _ElementTree, parse
from numpy import array, ndarray
from PIL import Image
from progress.bar import Bar

Corners = namedtuple("Corners", ["xmin", "ymin", "xmax", "ymax"], defaults=[0, 0, 0, 0])


def main() -> None:
    with open("annotationMap.json", "r") as jsonFile:
        data: dict = load(jsonFile)
        jsonFile.close()

    topLevelKeys: list = data.keys()

    with Bar("Checking bounding boxes for each type of aoi map...", max=10) as bar:
        tlk: str
        for tlk in topLevelKeys:
            xmlFilePath: str
            for xmlFilePath in data[tlk].keys():
                try:
                    et: _ElementTree = parse(xmlFilePath)
                except OSError:
                    continue

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
                            print("ERROR", xmlFilePath)

                topLeft: tuple = (
                    round(coordinates["xmin"]),
                    round(coordinates["ymin"]),
                )
                bottomRight: tuple = (
                    round(coordinates["xmax"]),
                    round(coordinates["ymax"]),
                )

                imageFilePath: str = data[tlk][xmlFilePath]
                image: ndarray = array(Image.open(imageFilePath).convert("RGB"))
                cv2.rectangle(
                    image, topLeft, bottomRight, color=(255, 0, 0), thickness=1
                )
                cv2.imshow("dark", image)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

            bar.next()


if __name__ == "__main__":
    main()
