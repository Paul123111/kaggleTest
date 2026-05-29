#
# SETUP
#
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime

from IPython.display import display, Markdown
pd.set_option('display.max_columns', None)
sns.set_style('darkgrid')

SEED = 2026

#
# IMPORTS
#
def import_data(location, weather_agg, df_zika, df_dengue, df_score, df_weather, debug=False):
    criteria = "location == @location"
    df_z = df_zika.query(criteria)
    df_d = df_dengue.query(criteria)
    df_s = df_score.query(criteria)
    df_w = df_weather.query(criteria)

    df_i = pd.concat([df_z, df_s], ignore_index=True)
    df_w = pd.merge_asof(df_w, df_i[['EW_start_date']],
                         left_on='date', right_on='EW_start_date', direction='backward')
    
    cols = df_w.columns.to_list()
    cols = cols[:2] + cols[-1:] + cols[2:-1]
    df_w = df_w[cols].copy()

    df_w_agg = df_w.groupby("EW_start_date").agg(weather_agg).reset_index()
    cols = [(f"{f}__{a}" if a else f"{f}") for f,a in df_w_agg.columns]
    df_w_agg.columns = cols
    
    if debug:
        print(f"Data imported for location {location}")

    criteria = "EW_start_date >= '2016-01-01'"
    df_z.query(criteria).to_feather(f"data/zika_{location}.feather")
    df_d.query(criteria).to_feather(f"data/dengue_{location}.feather")
    df_s.query(criteria).to_feather(f"data/score_{location}.feather")
    df_w_agg.to_feather(f"data/weather_{location}.feather")

#
# MODEL BUILDING
#