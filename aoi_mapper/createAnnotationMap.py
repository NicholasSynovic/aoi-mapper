from json import dump
from os import listdir
from os.path import join
from pathlib import PurePath

from pandas import DataFrame
from progress.bar import Bar


def readDirectory(dir: PurePath) -> list:
    files: list = listdir(dir)
    filepaths: list = [join(dir, f) for f in files]
    return filepaths


def main() -> None:
    imageFiles: list = []
    imageFolders: list = ["bgRemoval", "depth", "saliency", "segmentation"]

    xmlPath: PurePath = PurePath("annotations")
    xmlFiles: list = readDirectory(dir=xmlPath)

    annotationMap: dict = {file: [] for file in xmlFiles}
    keys: list = annotationMap.keys()

    folder: str
    for folder in imageFolders:
        for foo in readDirectory(dir=PurePath(folder)):
            imageFiles.append(foo)

    with Bar(
        "Creating mapping between xml files and image files...", max=len(keys)
    ) as bar:
        key: str
        for key in keys:
            keyPurePath: PurePath = PurePath(key)
            query: str = keyPurePath.with_suffix("").name

            count = 0
            filepath: PurePath
            for filepath in imageFiles:
                filepathPurePath: PurePath = PurePath(filepath)
                if filepathPurePath.name.find(query) != -1:
                    annotationMap[key].append(filepath)
            bar.next()

    dump(annotationMap, open("annotationMap.json", "w"))


if __name__ == "__main__":
    main()
