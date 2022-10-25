from os import listdir
from os.path import join
from pathlib import PurePath

from pandas import DataFrame
from progress.bar import Bar


def readDirectory(dir: PurePath) -> list:
    files: list = listdir(dir)
    filepaths: list = [PurePath(join(dir, f)) for f in files]
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
        key: PurePath
        for key in keys:
            query: str = key.with_suffix("").name

            filepath: PurePath
            for filepath in imageFiles:
                if filepath.name.find(query) > -1:
                    annotationMap[key].append(filepath)
            bar.next()

    df: DataFrame = DataFrame(annotationMap)
    df.T.to_json("annotationMap.json")


if __name__ == "__main__":
    main()
