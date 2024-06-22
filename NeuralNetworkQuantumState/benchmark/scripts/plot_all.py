
import os, sys
import re
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import numpy as np
from math import sqrt
from matplotlib import rc
from matplotlib.font_manager import FontProperties

# rc('text', usetex=True)
# font_path = '/ssd/home/ustc1/.fonts/Times New Roman 400.ttf'
# my_font = FontProperties(fname=font_path)
plots_for_paper = False
if plots_for_paper:
    plt.rcParams['font.family'] = ['Times New Roman']

KEY_WORD='EMEAN'
KEY_WORD='eloc_mean'

# https://matplotlib.org/stable/users/explain/colors/colormaps.html
# color_maps = ['Pastel1', 'Pastel2', 'Paired', 'Accent', 'Dark2',
#               'Set1', 'Set2', 'Set3', 'tab10', 'tab20', 'tab20b', 'tab20c']
def construct_color_maps(num_colors=20, cmap_name="Paired"):
    color_mapping = []
    cmap = mpl.colormaps.get_cmap(cmap_name)
    for i in range(num_colors):
        color_mapping.append(cmap(i))
    return color_mapping

def format_title(name):
    subscript_mapping = {
        '0': '₀', '1': '₁', '2': '₂', '3': '₃', '4': '₄',
        '5': '₅', '6': '₆', '7': '₇', '8': '₈', '9': '₉'
    }
    for num, sub in subscript_mapping.items():
        name = name.replace(num, sub)
    return name

def read_ref_data(ref_fname="ref.fci"):
    print(f"ref_fname: {ref_fname}")
    ref_data = {}
    with open(ref_fname, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                system_name, value = parts
                ref_data[system_name.lower()] = float(value)
    return ref_data

# define your specicial color mapping
color_mapping = {
    "bas": "#BFBF00",
    "mcmc": "#1EB0A8"
    # "adam": "red",
    # "sophia": "blue",
    # "lbfgs": "green"
    # "benchmark":"orange",
    # "base-gnorm":"green",
    # "base-gnormv2": "yellow",
    # "base-glnorm": "peru",
}

def load_data_from_file(results_dir, ref_data, exclude_keys=["l1norm"], include_keys=None):
    plots = {}
    for filename in os.listdir(results_dir):
        is_exclude = False
        for _key in exclude_keys:
            if _key in filename:
                is_exclude = True
                break
        if is_exclude:
            continue

        if include_keys is not None:
            is_using = False
            for _key in include_keys:
                if _key in filename:
                    is_using = True
                    break
            if is_using == False:
                continue

        if filename.startswith("result-"):
            _, system_name, suffix = re.split(r'[_-]', filename, maxsplit=2)
            system_name = system_name.lower()

            print(f'system_name: {system_name} suffix: {suffix}')
            if system_name in ref_data:
                e_mean_values = []
                with open(os.path.join(results_dir, filename), 'r') as f:
                    for line in f:
                        # match = re.search(r'eloc_mean: (-?\d+\.\d+)', line)
                        match = re.search(KEY_WORD+': (-?\d+\.\d+)', line)
                        # print(f"match: {match}")
                        if match:
                            e_mean_values.append(float(match.group(1)))

                    differences = [abs(value - ref_data[system_name]) for value in e_mean_values]
                    #differences = [max(diff, 1e-11) for diff in differences]
                    smoothed_differences = pd.Series(differences).rolling(window=20).mean().tolist()
                    #smoothed_differences = differences
                    print(f"ref_fci: {ref_data[system_name]}")

                    if system_name not in plots:
                        plots[system_name] = {}
                    plots[system_name][suffix] = smoothed_differences
    return plots

def plot_all(datas, pic_dir="pic", out_fname="mol", figsize=(20, 10), legend_name_dict={}, max_data_len=30000):
    if not os.path.exists(pic_dir):
        os.makedirs(pic_dir)

    plt.figure(figsize=figsize)
    N = len(datas.items())
    row = int(sqrt(N))
    col = -(-N//row)

    index = 0
    for system_name in sorted(datas.keys()):
        suffixs_data = datas[system_name]
        print(f"ploting system={system_name}")
        index += 1
        plt.subplot(row, col, index)
        ci = 0
        # for suffix in ["benchmark", "amp", "ampv2"]:
        for suffix in sorted(suffixs_data.keys()):
            data = suffixs_data[suffix]
            c = color_mapping.get(suffix) if color_mapping.get(suffix) else color_mapping_list[ci]; ci+=1
            # plt.plot(data[:max_data_len], label=suffix, color=c)
            # plt.plot(data[:max_data_len], label=suffix.upper(), color=c)
            label = legend_name_dict[suffix] if legend_name_dict.get(suffix) else suffix.upper()
            plt.plot(data[:max_data_len], label=label, color=c)
        plt.axhline(y=0.0016, color='red', linestyle='--', label='chemical accuracy')
        #plt.xlim((0,2000))
        plt.xlabel("Epoches", fontsize='large')
        plt.ylabel("Absolute error(Ha)", fontsize='large')
        formatted_title = format_title(corrected_system_names.get(system_name, system_name))
        plt.title(formatted_title, fontsize='xx-large')
        plt.yscale("log")
        plt.legend()
        # plt.grid(True, which="both", linestyle="--", linewidth=0.5)

    plt.tight_layout(pad=0.5)
    save_fname = os.path.join(pic_dir, out_fname)
    print(f"save into {save_fname}")
    plt.savefig(save_fname, dpi=400)

# define your output file name
out_fname = "amp-diff.png"

# define reference energy file path
ref_fname = "ref.fci"

# define your figure legend names
legend_name_dict={"amp": "AMP+Sampling", "ampv2": "AMP+(Amplitude+Phase)", "benchmark": "FP32"}

results_dir = sys.argv[1] if len(sys.argv) == 2 else "./result"
print(f"read data from {results_dir}")

color_mapping_list = construct_color_maps()
ref_data = read_ref_data(ref_fname)
corrected_system_names = {name.lower(): name for name in ref_data.keys()}
datas = load_data_from_file(results_dir, ref_data)
plot_all(datas, pic_dir="pic", out_fname=out_fname, figsize=(20, 10), legend_name_dict=legend_name_dict, max_data_len=30000)
