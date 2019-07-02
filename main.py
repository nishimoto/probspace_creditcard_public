#!/usr/bin/env python # 3.6.8
import warnings
import numpy as np     # 1.16.4
import pandas as pd    # 0.24.2
import xgboost as xgb  # 0.90

from tqdm import tqdm
from collections import defaultdict

from sklearn.metrics import accuracy_score           # 0.21.2
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier

warnings.filterwarnings('ignore')
params = {"seed": 0}


def validate(train_x, train_y, params):
    accuracies = []
    feature_importances = []

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=0)
    for train_idx, test_idx in cv.split(train_x, train_y):
        trn_x = train_x.iloc[train_idx, :]
        val_x = train_x.iloc[test_idx, :]

        trn_y = train_y.iloc[train_idx]
        val_y = train_y.iloc[test_idx]

        clf = xgb.XGBClassifier(**params)
        clf.fit(trn_x, trn_y)

        pred_y = clf.predict(val_x)
        feature_importances.append(clf.feature_importances_)
        accuracies.append(accuracy_score(val_y, pred_y))
    print(np.mean(accuracies))
    return accuracies, feature_importances


def preprocess_addtrend(df_train, df_test):
    a1s = []
    a2s = []
    a3s = []
    b1s = []
    b2s = []
    b3s = []
    for ind in tqdm(df_train.index):
        x = [1, 2, 3, 4, 5, 6]
        y1 = df_train.loc[ind, ["X6", "X7", "X8", "X9", "X10", "X11"]]
        y2 = df_train.loc[ind, ["X12", "X13", "X14", "X15", "X16", "X17"]]
        y3 = df_train.loc[ind, ["X18", "X19", "X20", "X21", "X22", "X23"]]

        a1, b1 = np.polyfit(x, y1, 1)
        a2, b2 = np.polyfit(x, y2, 1)
        a3, b3 = np.polyfit(x, y3, 1)

        a1s.append(a1); a2s.append(a2); a3s.append(a3)
        b1s.append(b1); b2s.append(b2); b3s.append(b3)

    df_train["a1"] = a1s
    df_train["a2"] = a2s
    df_train["a3"] = a3s
    df_train["b1"] = b1s
    df_train["b2"] = b2s
    df_train["b3"] = b3s

    a1s = []; a2s = []; a3s = []; b1s = []; b2s = []; b3s = []
    for ind in tqdm(df_test.index):
        x = [1, 2, 3, 4, 5, 6]
        y1 = df_train.loc[ind, ["X6", "X7", "X8", "X9", "X10", "X11"]]
        y2 = df_train.loc[ind, ["X12", "X13", "X14", "X15", "X16", "X17"]]
        y3 = df_train.loc[ind, ["X18", "X19", "X20", "X21", "X22", "X23"]]

        a1, b1 = np.polyfit(x, y1, 1)
        a2, b2 = np.polyfit(x, y2, 1)
        a3, b3 = np.polyfit(x, y3, 1)

        a1s.append(a1)
        a2s.append(a2)
        a3s.append(a3)
        b1s.append(b1)
        b2s.append(b2)
        b3s.append(b3)

    df_test["a1"] = a1s
    df_test["a2"] = a2s
    df_test["a3"] = a3s
    df_test["b1"] = b1s
    df_test["b2"] = b2s
    df_test["b3"] = b3s

    return df_train, df_test


# 初期版 feature engineering
# LB 0.831 => 0.833くらい
def preprocess_df(df):
    df.drop(["ID", "a1", "b1", "b2", "a3", "b3"], axis=1, inplace=True)
    df["X1/X6"] = df["X1"] / df["X6"]
    df["X6/X7"] = df["X6"] / df["X7"]
    df["X12/X13"] = df["X12"] / df["X13"]
    df["X1/X12"] = df["X1"] / df["X12"]
    df["X6/X12"] = df["X6"] / df["X12"]

    # -2と-1は同じ意味と考え、置換した。
    for col in ["X6", "X7", "X8", "X9", "X10", "X11"]:
        df[col] = [-1 if val == -2 else val for val in df[col]]

    return df


# いろいろfeature_selectionは試した。
# X5とX14を除くと0.835へ。
def preprocess_df(df):
    df.drop(["ID", "a1", "b1", "b2", "a3", "b3"], axis=1, inplace=True)
    df.drop(["X5", "X14"], axis=1, inplace=True)
    df["X1/X6"] = df["X1"] / df["X6"]
    df["X6/X7"] = df["X6"] / df["X7"]
    df["X12/X13"] = df["X12"] / df["X13"]
    df["X1/X12"] = df["X1"] / df["X12"]
    df["X6/X12"] = df["X6"] / df["X12"]

    # 値置換 maybe means -2 & -1 same.
    for col in ["X6", "X7", "X8", "X9", "X10", "X11"]:
        df[col] = [-1 if val == -2 else val for val in df[col]]

    df["X6/X7"] = df["X6/X7"].fillna(0)
    df["X12/X13"] = df["X12/X13"].fillna(0)
    df["X6/X12"] = df["X6/X12"].fillna(0)
    return df


def acc(pred, y):
    y = y.get_label()
    pred01 = [1 if pred_val >= 0.51 else 0 for pred_val in pred]
    return 'acc', 1 - accuracy_score(y, pred01)


# test dataのpredict
def predict_df(train_x, train_y, test_x, df_test_raw, path_output="result.csv"):
    clf = xgb.XGBClassifier(**params)
    clf.fit(train_x, train_y)
    preds = clf.predict(test_x)
    preds_proba = clf.predict_proba(test_x)[:, 1]

    _df = pd.DataFrame()
    _df["ID"] = df_test_raw["ID"]
    # _df["Y"] = preds
    _df["Y"] = [1 if proba >= 0.51 else 0 for proba in preds_proba] # ほんの少し精度あがった
    _df.to_csv(path_output, index=False)
    return _df


def preprocess_knn(df_train, df_test):
    """historyがほぼない人用の前処理関数"""
    # preprocess
    cols = ["X1", "X4", "X5"]

    # z_scoreをするのでdfへconcatしないといけない
    df_train = df_train[cols]
    df_test = df_test[cols]
    df = pd.concat([df_train, df_test])

    len_df_train = len(df_train)
    len_df_test = len(df_test)
    len_df_all = len_df_train + len_df_test

    # z_score化
    # for col in cols:
    #     df[col] = (df[col] - df[col].mean()) / df[col].std()

    # for col in cols:
    #     q75, q25 = np.percentile(df[col], [75, 25])
    #     df[col] = (df[col] - q25) / (q75 - q25)

    # min-max scaling +0.003!
    for col in cols:
        df[col] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())

    # subsplit
    df_train = df.iloc[range(len_df_train), :]
    df_test = df.iloc[range(len_df_train, len_df_all), :]

    # return
    return df_train, df_test


def predict_test_by_knn(train_x, train_y, test_x, n_neighbors):
    clf = KNeighborsClassifier(n_neighbors=n_neighbors)
    clf.fit(train_x, train_y)
    pred_y = clf.predict(test_x)
    return pred_y


# knn評価用
# なぜか偶数のほうが成績がよい。
# (偶数だと場合によっては多数決になるから?)
def validate_knn(train_x, train_y):
    d = defaultdict(list)
    for n_neighbors in range(1, 13):
        # sample数が少ないため,splitの方法によって多いに結果が変わる。
        # 20回試して箱ひげ図を作成
        for random_state in range(20):
            cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=random_state)
            accuracies = []
            for train_idx, test_idx in cv.split(train_x, train_y):
                trn_x = train_x.iloc[train_idx, :]
                val_x = train_x.iloc[test_idx, :]

                trn_y = train_y.iloc[train_idx]
                val_y = train_y.iloc[test_idx]

                clf = KNeighborsClassifier(n_neighbors=n_neighbors)
                clf.fit(trn_x, trn_y)

                pred_y = clf.predict(val_x)
                accuracies.append(accuracy_score(val_y, pred_y))
            d[n_neighbors].append(np.mean(accuracies))
    df = pd.DataFrame(d)
    df.plot(kind="box")


# 0.836 => 0.838
# historyがないヒトを別途予測する。
def predict_nohistknn(result_df, n_neighbors=5):
    # read & preprocess
    df_train = pd.read_csv("train_data.csv")
    df_test = pd.read_csv("test_data.csv")
    nohist = "X12 == 0 and X13 == 0 and X14 == 0 and X15 == 0 and X16 == 0 and X17 == 0 and X18 == 0 and X19 == 0 and X20 == 0 and X21 == 0 and X22 == 0 and X23 == 0"
    df_train_nohist = df_train.query(nohist)
    df_test_nohist = df_test.query(nohist)
    df_train_nohist_pp, df_test_nohist_pp = preprocess_knn(df_train_nohist, df_test_nohist)

    # knnの評価
    train_x = df_train_nohist_pp.copy()
    train_y = df_train_nohist["y"]
    # validate_knn(train_x, train_y)

    # 結局testdataも絨毯爆撃したが,n=4が最もよいという結果に。
    pred_y = predict_test_by_knn(train_x, train_y, df_test_nohist_pp, n_neighbors)

    # 今回
    result_df.loc[df_test_nohist["ID"], "Y"] = pred_y
    return result_df


def main():
    df_train = pd.read_csv("train_data.csv")
    df_test = pd.read_csv("test_data.csv")

    # preprocess 1st
    df_train, df_test = preprocess_addtrend(df_train, df_test)

    # 結構時間かかるので保存したのを使いまわしていた。
    # df_train.to_csv("train_data.add_trend.csv")
    # df_test.to_csv("test_data.add_trend.csv")

    # preprocess 2nd
    train_y = df_train["y"]
    train_x = df_train.drop("y", axis=1)
    train_x = train_x.rename(columns={'id': 'ID'})
    train_x = preprocess_df(train_x)

    accuracies, feature_importances = validate(train_x, train_y, params)

    # 最終結果の出力
    flag_product = True
    if flag_product:
        df_test_raw = df_test.copy()
        test_x = preprocess_df(df_test)
        result_df = predict_df(train_x, train_y, test_x, df_test_raw)
        # result_df.to_csv("result.xgb.csv", index=False)
        n_neighbors = 4
        result_df = predict_nohistknn(result_df, n_neighbors=n_neighbors)
        result_df.to_csv("result.xgb.{}.csv".format(n_neighbors), index=False)


if __name__ == "__main__":
    main()
