import pandas as pd
from pyecharts import options as opts
from pyecharts.charts import Kline
from snapshot_selenium import snapshot as driver
import numpy as np
from pyecharts import options as opts
from pyecharts.charts import Bar
from pyecharts.render import make_snapshot
import util.constant


def generate_data(data_dir: str):
    """
    用于生成图片 以产生label
    :param data_dir:
    :return:
    """
    data = pd.read_csv(data_dir)
    windows_size = util.constant.WINDOW_SIZE
    feature_nums = util.constant.FEATURE_NUMS
    train_num = int(len(data) * 0.7)
    train_data = data.iloc[0:train_num, :]
    valid_num = int(len(data) * 0.2)
    valid_data = data.iloc[train_num:train_num + valid_num, :]
    test_data = data.iloc[train_num + valid_num:, :]
    train_y = []
    # label产生
    y = train_data.iloc[windows_size - 1:, train_data.shape[1] - 1:train_data.shape[1]]
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
    for i in range(len(train_data) - windows_size):
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
        name = "F:\\project\\sp500\\data\\train\\{}\\bar_{}.png".format(sub_name, i)
        make_snapshot(driver, k.render(), name)


if __name__ == '__main__':
    generate_data(r"F:\project\sp500\data\SP500.csv")
