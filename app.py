#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 27 10:35:26 2025

@author: zha
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.spatial import Voronoi, voronoi_plot_2d
from sklearn.preprocessing import KBinsDiscretizer
from scipy.optimize import curve_fit
from jenkspy import JenksNaturalBreaks
import streamlit as st
from shapely.geometry import Polygon
from shapely.ops import unary_union

# ==============================
# 1. 随机生成聚落数据
# ==============================
np.random.seed(42)
N = 50  # 城市/聚落数量

x = np.random.uniform(0, 100, N)
y = np.random.uniform(0, 100, N)
population = np.random.lognormal(mean=10, sigma=0.5, size=N).astype(int)

cities = pd.DataFrame({"x": x, "y": y, "pop": population})

# ==============================
# 2. 自动分层函数
# ==============================
def classify_levels(data, method="quantile", n_bins=3):
    if method == "quantile":
        discretizer = KBinsDiscretizer(n_bins=n_bins, encode="ordinal", strategy="quantile")
        levels = discretizer.fit_transform(data[["pop"]]).astype(int).ravel()
    elif method == "uniform":
        discretizer = KBinsDiscretizer(n_bins=n_bins, encode="ordinal", strategy="uniform")
        levels = discretizer.fit_transform(data[["pop"]]).astype(int).ravel()
    elif method == "jenks":
        jnb = JenksNaturalBreaks(n_classes=n_bins)  # ⚠️ 修复
        jnb.fit(data["pop"].values)
        levels = jnb.labels_
    else:
        raise ValueError("Unknown method, choose quantile/uniform/jenks")
    return levels

# ==============================
# 3. 绘图函数
# ==============================

def plot_model(method="jenks", n_bins=3):
    cities["level"] = classify_levels(cities, method=method, n_bins=n_bins)

    # Voronoi 服务区
    points = cities[["x", "y"]].values
    vor = Voronoi(points)

    fig, ax = plt.subplots(figsize=(8, 8))

    # 定义颜色映射
    color_map = {0:"green", 1:"blue", 2:"red", 3:"orange", 4:"purple"}
    labels = {0:"小城镇", 1:"中城市", 2:"大城市", 3:"超大城市", 4:"特大城市"}

    # 绘制 Voronoi 多边形并填充颜色
    for point_idx, region_idx in enumerate(vor.point_region):
        vertices = vor.regions[region_idx]
        if -1 in vertices or len(vertices) == 0:
            continue  # 忽略无界区域
        polygon = Polygon(vor.vertices[vertices])
        if polygon.is_valid:
            lvl = cities.loc[point_idx, "level"]
            ax.fill(*polygon.exterior.xy, 
                    color=color_map[lvl % len(color_map)], 
                    alpha=0.4, 
                    edgecolor="black")

    # 绘制城市点
    for _, row in cities.iterrows():
        ax.scatter(row["x"], row["y"], 
                   s=row["pop"]*0.0005, 
                   c=color_map[row["level"] % len(color_map)], 
                   edgecolor="k", alpha=0.8)
        ax.text(row["x"]+0.8, row["y"]+0.8, 
                f"{labels[row['level']]}\nPop:{row['pop']}",
                fontsize=8, color=color_map[row["level"] % len(color_map)])

    ax.set_title(f"Central Place Simulation ({method}, {n_bins} bins)")
    ax.set_xlabel("X coordinate")
    ax.set_ylabel("Y coordinate")

    return fig


# ==============================
# 4. Streamlit 界面
# ==============================
st.title("Central Place Theory Simulation")

method = st.sidebar.selectbox("Classification Method", ["quantile", "uniform", "jenks"])
n_bins = st.sidebar.slider("Number of Bins", 2, 6, 3)

fig1, fig2, stats = plot_model(method, n_bins)

# ==============================
# 布局
# ==============================

st.pyplot(fig1)
st.subheader("Level Statistics")
st.dataframe(stats)

st.pyplot(fig2)

# ==============================
# 固定右下角标注
# ==============================
st.markdown(
    """
    <style>
    .footer {
        position: fixed;
        right: 10px;
        bottom: 10px;
        font-size: 12px;
        color: grey;
        opacity: 0.8;
    }
    </style>
    <div class="footer">
        Produced by Centre for Urban Science and Planning, Tongji
    </div>
    """,
    unsafe_allow_html=True
)
