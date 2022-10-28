from pandas import DataFrame
import pandas
from numpy import ndarray

def main()  ->  None:
    data: dict = {}

    df: DataFrame = pandas.read_json("bboxStatistics.zip", compression="zip").T
    columns: ndarray = df.columns
    aoiMethods: ndarray = df["aoiMethod"].unique()

    subDFs: list = [df[df["aoiMethod"] == method] for method in aoiMethods]

    sub: DataFrame
    for sub in subDFs:
        method: str = sub["aoiMethod"].unique()[0]
        data[method] = []

        data[method] = sub["bboxWhitePixelPercentage"].mean()
        data[method] = sub["bboxBlackPixelPercentage"].mean()

    data.__delitem__("bbox")
    print(data)

if __name__ == "__main__":
    main()
