
import pandas as pd


if __name__ == '__main__':

    det_result = [
        [5, 0.9, 0.5, 0.5, 0.2, 0.3],   # [cls, con, x, y, w, h]
        [2, 0.8, 0.1, 0.4, 0.5, 0.36],
        [0, 0.91, 0.35, 0.85, 0.24, 0.32],
        [6, 0.59, 0.25, 0.35, 0.22, 0.31],
        [6, 0.6, 0.15, 0.51, 0.12, 0.33],
    ]

    df = pd.DataFrame(det_result, columns=['cls', 'con', 'x', 'y', 'w', 'h'])

    print(df)

    df.to_excel("./output.xlsx")
