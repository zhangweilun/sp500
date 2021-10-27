import pandas as pd
from pyecharts import options as opts
from pyecharts.charts import Kline
from snapshot_selenium import snapshot as driver
import numpy as np
from pyecharts import options as opts
from pyecharts.charts import Bar
from pyecharts.render import make_snapshot
import util.constant as constant


def generate_data(data_dir: str, env: constant.Env) -> pd.DataFrame:
    data = pd.read_csv(data_dir)
    train_num = int(len(data) * 0.7)
    train_data = data.iloc[0:train_num, :]
    valid_num = int(len(data) * 0.2)
    valid_data = data.iloc[train_num:train_num + valid_num, :]
    test_data = data.iloc[train_num + valid_num:, :]
    if env.TRAIN == env:
        return train_data
    elif env.VALID == env:
        return valid_data
    elif env.TEST == env:
        return test_data


def out_image(out_dir: str, data: pd.DataFrame):
    """
    用于生成图片 以产生label
    :param data:
    :param out_dir: 输出目录
    :return:
    """
    windows_size = constant.WINDOW_SIZE
    feature_nums = constant.FEATURE_NUMS
    train_y = []
    # label产生
    y = data.iloc[windows_size - 1:, data.shape[1] - 1:data.shape[1]]
    # 相邻两行相减
    y["tump"] = y["y"].shift(1)
    gap_y = y["y"] - y["tump"]
    for i, v in gap_y.items():
        # print(i, v)
        if pd.isna(v):
            continue
        if v >= 0:
            train_y.append(1)
        else:
            train_y.append(0)
    # 产生训练数据
    for i in range(len(data) - windows_size):
        # open close lowest highest
        x = data.iloc[i:i + windows_size, 1:feature_nums]
        time = data.iloc[i:i + windows_size, :1]
        time = np.array(time).squeeze(axis=1)
        time = time.tolist()
        x = np.array(x).tolist()
        itemstyle_opts = opts.ItemStyleOpts(color="#ec0000", color0="#00da3c", border_color="#8A0000",
                                            border_color0="#008F28")
        xaxis_opts = opts.AxisOpts(is_scale=True)
        yaxis_opts = opts.AxisOpts(is_scale=True, splitarea_opts=opts.SplitAreaOpts(is_show=True,
                                                                                    areastyle_opts=opts.AreaStyleOpts(
                                                                                        opacity=1)), )
        title_opts = opts.TitleOpts(title="Kline-ItemStyle")
        k = Kline()
        k.add_xaxis(time)
        k.add_yaxis("Kline", x, itemstyle_opts=itemstyle_opts)
        k.set_global_opts(xaxis_opts=xaxis_opts, yaxis_opts=yaxis_opts, title_opts=title_opts)
        sub_name = ""
        if train_y[i] == 0:
            sub_name = "dog"
        elif train_y[i] == 1:
            sub_name = "cat"
        name = out_dir + r"\{}\bar_{}.png".format(sub_name, i)
        make_snapshot(driver, k.render(), name)


if __name__ == '__main__':
    data = generate_data(r"F:\project\sp500\data\SP500.csv", constant.Env.TRAIN)
    out_image(r"F:\project\sp500\data" + constant.Env.TRAIN.name, data)
