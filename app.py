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

    color_map = {0:"green", 1:"blue", 2:"red", 3:"orange", 4:"purple"}
    
    # ------- 图 1：Voronoi 地图 -------
    fig1, ax1 = plt.subplots(figsize=(8,8))
    voronoi_plot_2d(vor, ax=ax1, show_vertices=False, show_points=False)

    colors = ["green", "blue", "red", "orange", "purple", "brown"]
    for i, row in cities.iterrows():
        ax1.scatter(row["x"], row["y"], s=row["pop"]*0.0008, 
                    c=colors[int(row["level"]) % len(colors)], alpha=0.6)
        ax1.text(row["x"]+0.8, row["y"]+0.8, f"Pop:{row['pop']}", fontsize=7)

    ax1.set_title(f"Central Place Simulation ({method}, {n_bins} bins)")
    #ax1.set_xlabel("X coordinate")
    #ax1.set_ylabel("Y coordinate")

    # ------- 动态图例 -------
    unique_levels = sorted(cities["level"].unique())  # 当前实际出现的层级
    labels = {0: "Small Town", 1: "Medium City", 2: "Large City",
              3: "Regional Hub", 4: "Metropolis"}
    
    for lvl in unique_levels:
        ax1.scatter([], [], color=color_map[lvl % len(color_map)], 
                    alpha=0.6, label=labels.get(lvl, f"Level {lvl}"))

    ax1.legend(
        title="City Level",
        loc="upper left", 
        bbox_to_anchor=(1.02, 1), 
        borderaxespad=0
    )
    #ax1.legend(title="City Level", loc="upper right")

    # ------- 计算 Voronoi 区域面积 -------
    regions = {}
    for point_idx, region_idx in enumerate(vor.point_region):
        vertices = vor.regions[region_idx]
        if -1 in vertices or len(vertices) == 0:
            continue
        polygon = Polygon(vor.vertices[vertices])
        if polygon.is_valid:
            regions[point_idx] = polygon.area

    cities["area"] = cities.index.map(regions).fillna(np.nan)

    stats = cities.groupby("level").agg(
        Avg_Pop=("pop", "mean"),
        Avg_Area=("area", "mean"),
        City_Count=("pop", "count")
    )

    # ------- 图 2：Rank-Size 拟合 -------
    cities_sorted = cities.sort_values("pop", ascending=False).reset_index(drop=True)
    cities_sorted["rank"] = np.arange(1, len(cities_sorted)+1)

    def rank_size(r, P1, q):
        return P1 / (r**q)

    popt, _ = curve_fit(rank_size, cities_sorted["rank"], cities_sorted["pop"], 
                        p0=[cities_sorted["pop"].iloc[0], 1])

    fig2, ax2 = plt.subplots(figsize=(8,6))
    ax2.scatter(cities_sorted["rank"], cities_sorted["pop"], label="Data")
    ax2.plot(cities_sorted["rank"], rank_size(cities_sorted["rank"], *popt), 
             'r--', label=f"Fit: q={popt[1]:.2f}")
    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.set_xlabel("Rank")
    ax2.set_ylabel("Population")
    ax2.set_title("Rank-Size Rule")
    ax2.legend()

    return fig1, fig2, stats


# ==============================
# 4. Streamlit 界面
# ==============================
st.title("Central Place Theory Simulation")

method = st.sidebar.selectbox("Classification Method", ["quantile", "uniform", "jenks"])
n_bins = st.sidebar.slider("Number of Bins", 2, 5, 3)

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
