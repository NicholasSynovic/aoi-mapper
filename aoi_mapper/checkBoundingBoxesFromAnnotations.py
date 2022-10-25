from collections import namedtuple
from json import load

from lxml.etree import _Element, _ElementTree, parse
from progress.bar import Bar

Corners = namedtuple("Corners", ["xmin", "ymin", "xmax", "ymax"])


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

                coordinates: Corners = Corners(
                    bndBoxElement[0].text,
                    bndBoxElement[1].text,
                    bndBoxElement[2].text,
                    bndBoxElement[3].text,
                )

                quit()

            bar.next()


if __name__ == "__main__":
    main()
