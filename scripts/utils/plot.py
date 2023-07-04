import seaborn as sb
import matplotlib.pyplot as plt
import pandas
import os
import argparse
from matplotlib.ticker import FormatStrFormatter
from matplotlib.ticker import MaxNLocator
from matplotlib import rc

PLOT_FONT_SIZE = 22
PLOT_LEGEND_SIZE = 18
PLOT_TICKS_SIZE = 18
PLOT_LINE_WIDTH = 4

plt.rcParams["figure.figsize"] = [12, 8]
plt.rcParams["figure.autolayout"] = True
sb.set_palette("bright")


def main():
    args = plot_args()
    dataframe_path = args.dataframe_path
    os.path.exists(dataframe_path), "dataframe does not exist!"
    plot(dataframe_path)


def plot(dataframe_path):
    # DTU experiment plot
    save_path = os.path.dirname(dataframe_path)
    total_df = pandas.read_csv(dataframe_path)

    fig, (ax1, ax2) = plt.subplots(1, 2, sharex=True)
    sb.lineplot(
        total_df,
        x="Reference Image Number",
        y="PSNR",
        hue="Planning Type",
        linewidth=PLOT_LINE_WIDTH,
        ax=ax1,
        errorbar=("sd", 1),
        palette=["C2", "C0", "C3"],
    )

    ax1.set_ylabel("PSNR", fontsize=PLOT_FONT_SIZE)
    ax1.set_xlabel("Number of collected images", fontsize=PLOT_FONT_SIZE)
    ax1.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    ax1.yaxis.set_major_locator(MaxNLocator(nbins=4))
    ax1.tick_params(axis="both", labelsize=PLOT_TICKS_SIZE)

    handles, labels = ax1.get_legend_handles_labels()
    order = [2, 0, 1]
    ax1.legend(
        [handles[idx] for idx in order],
        [labels[idx] for idx in order],
        loc="lower right",
        fontsize=PLOT_LEGEND_SIZE,
        frameon=False,
    )

    sb.lineplot(
        total_df,
        x="Reference Image Number",
        y="SSIM",
        hue="Planning Type",
        linewidth=PLOT_LINE_WIDTH,
        ax=ax2,
        errorbar=("sd", 1),
        palette=["C2", "C0", "C3"],
    )
    ax2.set_xlabel("Number of collected images", fontsize=PLOT_FONT_SIZE)
    ax2.set_ylabel("SSIM", fontsize=PLOT_FONT_SIZE)
    ax2.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    ax2.yaxis.set_major_locator(MaxNLocator(nbins=4))
    ax2.get_legend().remove()
    ax2.tick_params(axis="both", labelsize=PLOT_TICKS_SIZE)

    plt.xticks([2, 3, 4, 5, 6, 7, 8, 9])
    plt.savefig(f"{save_path}/plot_results.svg", bbox_inches="tight")
    plt.clf()


def plot_args():
    # mandatory arguments
    args = None
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataframe_path",
        type=str,
        required=True,
        help="path to dataframe",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    main()

# # gazebo car experiment plot
# total_df = pandas.read_csv("dataframe/experiment_gazebo_car_0.csv")
# total_df = total_df.append(pandas.read_csv("dataframe/experiment_gazebo_car_1.csv"))

# fig, (ax1, ax2) = plt.subplots(1, 2, sharex=True)
# sb.lineplot(
#     total_df,
#     x="Reference Image Num.",
#     y="PSNR",
#     hue="Planning Type",
#     linewidth=PLOT_LINE_WIDTH,
#     ax=ax1,
#     errorbar=("sd", 1),
#     palette=["C2", "C0", "C3"],
# )
# ax1.set_xlabel("Number of collected images", fontsize=PLOT_FONT_SIZE)
# ax1.set_ylabel("PSNR", fontsize=PLOT_FONT_SIZE)
# ax1.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
# ax1.yaxis.set_major_locator(MaxNLocator(nbins=4))
# # ax1.tick_params(labelsize=PLOT_TICKS_SIZE)
# ax1.legend(loc="lower right", fontsize=PLOT_FONT_SIZE)
# ax1.tick_params(axis="both", labelsize=PLOT_TICKS_SIZE)

# handles, labels = ax1.get_legend_handles_labels()
# order = [2, 0, 1]
# ax1.legend(
#     [handles[idx] for idx in order],
#     [labels[idx] for idx in order],
#     loc="lower right",
#     fontsize=PLOT_LEGEND_SIZE,
#     frameon=False,
# )

# sb.lineplot(
#     total_df,
#     x="Reference Image Num.",
#     y="SSIM",
#     hue="Planning Type",
#     linewidth=PLOT_LINE_WIDTH,
#     ax=ax2,
#     errorbar=("sd", 1),
#     palette=["C2", "C0", "C3"],
# )
# ax2.set_xlabel("Number of collected images", fontsize=PLOT_FONT_SIZE)
# plt.xticks([2, 4, 6, 8, 10, 12, 14, 16, 18, 20])
# ax2.set_ylabel("SSIM", fontsize=PLOT_FONT_SIZE)
# ax2.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
# ax2.yaxis.set_major_locator(MaxNLocator(nbins=4))
# ax2.get_legend().remove()
# ax2.tick_params(axis="both", labelsize=PLOT_TICKS_SIZE)
# # ax2.tick_params(labelsize=PLOT_TICKS_SIZE)
# # plt.show()
# plt.xticks(fontsize=PLOT_TICKS_SIZE)
# plt.yticks(fontsize=PLOT_TICKS_SIZE)
# plt.savefig("dataframe/gazebo_car.svg", bbox_inches="tight")
# plt.clf()

# # # gazebo indoor experiment plot
# total_df = pandas.read_csv("dataframe/experiment_gazebo_indoor_0.csv")
# total_df = total_df.append(pandas.read_csv("dataframe/experiment_gazebo_indoor_1.csv"))

# fig, (ax1, ax2) = plt.subplots(1, 2, sharex=True)
# sb.lineplot(
#     total_df,
#     x="Reference Image Num.",
#     y="PSNR",
#     hue="Planning Type",
#     linewidth=PLOT_LINE_WIDTH,
#     ax=ax1,
#     errorbar=("sd", 1),
#     palette=["C2", "C0", "C3"],
# )
# ax1.set_ylabel("PSNR", fontsize=PLOT_FONT_SIZE)
# ax1.set_xlabel("Number of collected images", fontsize=PLOT_FONT_SIZE)
# ax1.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
# ax1.set_yticks([13.4, 14.6, 15.8, 17.0])
# # ax1.yaxis.set_major_locator(MaxNLocator(nbins=5))
# ax1.tick_params(axis="both", labelsize=PLOT_TICKS_SIZE)
# # ax1.tick_params(labelsize=PLOT_TICKS_SIZE)
# ax1.legend(loc="lower right", fontsize=PLOT_FONT_SIZE)
# handles, labels = ax1.get_legend_handles_labels()
# order = [2, 0, 1]
# ax1.legend(
#     [handles[idx] for idx in order],
#     [labels[idx] for idx in order],
#     loc="lower right",
#     fontsize=PLOT_LEGEND_SIZE,
#     frameon=False,
# )

# sb.lineplot(
#     total_df,
#     x="Reference Image Num.",
#     y="SSIM",
#     hue="Planning Type",
#     linewidth=PLOT_LINE_WIDTH,
#     ax=ax2,
#     errorbar=("sd", 1),
#     palette=["C2", "C0", "C3"],
# )
# ax2.set_xlabel("Number of collected images", fontsize=PLOT_FONT_SIZE)
# ax2.set_ylabel("SSIM", fontsize=PLOT_FONT_SIZE)
# ax2.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
# ax2.yaxis.set_major_locator(MaxNLocator(nbins=4))
# ax2.get_legend().remove()
# ax2.tick_params(axis="both", labelsize=PLOT_TICKS_SIZE)
# # ax2.tick_params(labelsize=PLOT_TICKS_SIZE)
# # plt.show()
# plt.xticks([2, 4, 6, 8, 10, 12, 14, 16, 18, 20])
# plt.xticks(fontsize=PLOT_TICKS_SIZE)
# plt.yticks(fontsize=PLOT_TICKS_SIZE)
# plt.savefig("dataframe/gazebo_indoor.svg", bbox_inches="tight")
# plt.clf()
