import pandas as pd
from pyecharts import options as opts
from pyecharts.charts import Kline
from snapshot_selenium import snapshot as driver
import numpy as np
from pyecharts import options as opts
from pyecharts.charts import Bar
from pyecharts.render import make_snapshot
import util.constant

if __name__ == '__main__':
    data = pd.read_csv('./data/SP500.csv')
    windows_size = util.constant.WINDOW_SIZE
    feature_nums = util.constant.FEATURE_NUMS
    for i in range(len(data) - windows_size):
        # open close lowest highest
        x = data.iloc[i:i + windows_size, 1:feature_nums]
        time = data.iloc[i:i + windows_size, :1]
        time = np.array(time).squeeze(axis=1)
        time = time.tolist()
        x = np.array(x).tolist()
        # k = Kline()
        # k.add_xaxis(time)
        # k.add_yaxis("Kline", x)
        c = (
            Kline()
                .add_xaxis(time)
                .add_yaxis(
                "kline",
                x,
                itemstyle_opts=opts.ItemStyleOpts(
                    color="#ec0000",
                    color0="#00da3c",
                    border_color="#8A0000",
                    border_color0="#008F28",
                ),
            )
                .set_global_opts(
                xaxis_opts=opts.AxisOpts(is_scale=True),
                yaxis_opts=opts.AxisOpts(
                    is_scale=True,
                    splitarea_opts=opts.SplitAreaOpts(
                        is_show=True, areastyle_opts=opts.AreaStyleOpts(opacity=1)
                    ),
                ),
                # datazoom_opts=[opts.DataZoomOpts(type_="inside")],
                title_opts=opts.TitleOpts(title="Kline-ItemStyle"),
            )
        )
        name = "F:\\project\\sp500\\data\\bar_{}.png".format(i)
        # c.render("kline_itemstyle.html")
        make_snapshot(driver, c.render(), name)
