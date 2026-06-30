# %%
#import data
import retinanalysis as ra
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import pandas as pd
import os
from datetime import date


# %%
def get_experiment_date(stim_block):
    exp_name = stim_block.exp_name

    year = int(exp_name[0:4])
    month = int(exp_name[4:6])
    day = int(exp_name[6:8])

    return date(year, month, day)

# %% [markdown]
# ## Get spike counts (Reusable)

# %%
def get_spike_count_table(response_block, cell_types=None, minimum_n=3):
    spike_times = ra.get_spike_xarr(
        response_block,
        cell_types=cell_types,
        minimum_n=minimum_n
    )

    spike_counts = xr.apply_ufunc(
        len,
        spike_times,
        vectorize=True
    )

    spike_counts = spike_counts.to_dataframe(
        name="spike_count"
    ).reset_index()

    return spike_counts

# %% [markdown]
# ## Add parameter column (Reusable) Aligns epoch with constant values

# %%
def add_parameter_column(spike_counts, stim_block, parameter_name):
    values = stim_block.d_epoch_block_params[parameter_name]

    epoch_to_value = dict(enumerate(values))

    spike_counts[parameter_name] = spike_counts["epoch"].map(epoch_to_value)

    return spike_counts

# %% [markdown]
# ## Average responses by parameter (Reusable)

# %%
def average_response_by_parameter(spike_counts, parameter_name):
    summary = (
        spike_counts
        .groupby(["cell_id", parameter_name])["spike_count"]
        .mean()
        .reset_index()
    )

    return summary

# %% [markdown]
# ## Metric index 

# %%
def normalized_difference(low_response, high_response):
    return (high_response - low_response) / (
        high_response + low_response
    )

# %% [markdown]
# ## Apply metric to each cell

# %%
def calculate_metric_table(
    summary,
    parameter_name,
    metric_func,
    metric_name
):
    rows = []

    for cell_id in summary["cell_id"].unique():
        cell_data = summary[summary["cell_id"] == cell_id]

        smallest_value = cell_data[parameter_name].min()
        largest_value = cell_data[parameter_name].max()

        low_response = cell_data[
            cell_data[parameter_name] == smallest_value
        ]["spike_count"].values[0]

        high_response = cell_data[
            cell_data[parameter_name] == largest_value
        ]["spike_count"].values[0]

        metric_value = metric_func(low_response, high_response)

        rows.append({
            "cell_id": cell_id,
            "Smallest value": smallest_value,
            "Largest value": largest_value,
            "Response at smallest value": low_response,
            "Response at largest value": high_response,
            metric_name: metric_value
        })

    return pd.DataFrame(rows)

# %% [markdown]
# ## Final function

# %%
def create_general_response_table(
    pipe,
    parameter_name,
    metric_func,
    metric_name,
    cell_types=None,
    minimum_n=3
):
    response_block = pipe.response_block
    stim_block = pipe.stim_block

    exp_date = get_experiment_date(stim_block)

    spike_counts = get_spike_count_table(
        response_block,
        cell_types=cell_types,
        minimum_n=minimum_n
    )

    spike_counts = add_parameter_column(
        spike_counts,
        stim_block,
        parameter_name
    )

    summary = average_response_by_parameter(
        spike_counts,
        parameter_name
    )

    metric_table = calculate_metric_table(
        summary,
        parameter_name,
        metric_func,
        metric_name
    )

    metric_table["Date of experiment"] = exp_date
    metric_table["Parameter name"] = parameter_name

    metric_table = metric_table[
        [
            "cell_id",
            "Date of experiment",
            "Parameter name",
            "Smallest value",
            "Largest value",
            "Response at smallest value",
            "Response at largest value",
            metric_name
        ]
        ]

    return metric_table
    



