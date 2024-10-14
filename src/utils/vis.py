import math

import cv2
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.manifold import TSNE

from src.data.transform import NormalizeBbox, NormalizeKeypoints

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman"]

_cm_tab10 = plt.get_cmap("tab10")
_cm_jet = plt.get_cmap("jet", 100)


EDGE_INDEX = [
    (0, 1),  # Head
    (0, 2),
    (1, 3),
    (2, 4),
    (5, 6),  # Trunk
    (5, 11),
    (6, 12),
    (11, 12),
    (5, 7),  # Arm
    (7, 9),
    (6, 8),
    (8, 10),
    (11, 13),  # Leg
    (13, 15),
    (12, 14),
    (14, 16),
]


def draw_skeleton(
    frame: np.array, kps: np.array, color, thickness=2, plot_limbs_only=False
):
    part_line = {}

    # draw keypoints
    for n in range(len(kps)):
        cor_x, cor_y = int(kps[n, 0]), int(kps[n, 1])
        part_line[n] = (cor_x, cor_y)
        if not plot_limbs_only:
            cv2.circle(frame, (cor_x, cor_y), 3, color, 1)

    # draw limbs
    for start_p, end_p in EDGE_INDEX:
        if start_p in part_line and end_p in part_line:
            start_xy = part_line[start_p]
            end_xy = part_line[end_p]
            cv2.line(frame, start_xy, end_xy, color, thickness)

    return frame


def draw_bbox(frame: np.array, bbox: np.array, color: tuple, thickness=2):
    pt1, pt2 = bbox.astype(int).reshape(2, 2)
    frame = cv2.rectangle(frame, pt1, pt2, color, thickness)
    return frame


def plot_bbox_on_frame(frame, results, idx_data, frame_size, range_points):
    for data in results:
        _id = data["id"]
        bbox = data["bbox"][idx_data]
        mse_bbox = data["mse_bbox"]
        fake_bbox = data["recon_bbox"][idx_data]

        bbox = NormalizeBbox.reverse(bbox, frame_size, range_points)
        fake_bbox = NormalizeBbox.reverse(fake_bbox, frame_size, range_points)

        # id
        pt = tuple(np.mean(bbox, axis=0).astype(int))
        frame = cv2.putText(
            frame, str(_id), pt, cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2
        )

        # bbox and fake_bbox
        frame = draw_bbox(frame, bbox, color=(0, 255, 0))
        frame = draw_bbox(frame, fake_bbox, color=(0, 0, 255))

        # mse
        pt = tuple(np.min(bbox, axis=0).astype(int))  # top-left
        frame = cv2.putText(
            frame,
            f"{mse_bbox:.3f}",
            pt,
            cv2.FONT_HERSHEY_COMPLEX,
            0.8,
            (255, 255, 255),
            2,
        )

    return frame


def plot_kps_on_frame(frame, results, idx_data, frame_size, range_points):
    for data in results:
        _id = data["id"]
        bbox = data["bbox"][idx_data]
        kps = data["kps"][idx_data]
        fake_kps = data["recon_kps"][idx_data]
        mse_kps = data["mse_kps"]

        bbox = NormalizeBbox.reverse(bbox, frame_size, range_points)
        kps = NormalizeKeypoints.reverse(kps, bbox, range_points)
        fake_kps = NormalizeKeypoints.reverse(fake_kps, bbox, range_points)

        # id
        pt = tuple(np.mean(bbox, axis=0).astype(int))
        frame = cv2.putText(
            frame, str(_id), pt, cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2
        )

        # kps
        frame = draw_skeleton(frame, kps, color=(0, 255, 0))
        frame = draw_skeleton(frame, fake_kps, color=(0, 0, 255))

        # mse
        pt = tuple(np.min(bbox, axis=0).astype(int))  # top-left
        frame = cv2.putText(
            frame,
            f"{mse_kps:.3f}",
            pt,
            cv2.FONT_HERSHEY_COMPLEX,
            0.8,
            (255, 255, 255),
            2,
        )
    return frame


def plot_cluster_on_frame(frame, results, idx_data, frame_size, range_points):
    for data in results:
        _id = data["id"]
        bbox = data["bbox"][idx_data]
        label = str(data["label"])

        bbox = NormalizeBbox.reverse(bbox, frame_size, range_points)

        # id
        pt = tuple(np.mean(bbox, axis=0).astype(int))
        frame = cv2.putText(
            frame, str(_id), pt, cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2
        )

        # bbox
        color = (np.array(_cm_tab10(int(label))[:3]) * 255).astype(int).tolist()
        color = tuple(color[::-1])  # RGB -> BGR
        frame = draw_bbox(frame, bbox, color)

        # clustering label
        pt = tuple(np.min(bbox, axis=0).astype(int))  # top-left
        frame = cv2.putText(frame, label, pt, cv2.FONT_HERSHEY_COMPLEX, 0.8, color, 2)
    return frame


def plot_attention_on_frame(
    frame, results, idx_data, frame_size, range_points, vmax=0.5
):
    for data in results:
        label = str(data["label"])
        bbox = data["bbox"][idx_data]
        kps = data["kps"][idx_data]
        attn_w = data["attn_w"]

        bbox = NormalizeBbox.reverse(bbox, frame_size, range_points)
        kps = NormalizeKeypoints.reverse(kps, bbox, range_points)

        # clustering label
        pt = tuple(np.mean(bbox, axis=0).astype(int))
        frame = cv2.putText(
            frame, label, pt, cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2
        )

        # plot bbox and skeleton limbs
        color = (np.array(_cm_tab10(int(label))[:3]) * 255).astype(int).tolist()
        color = tuple(color[::-1])  # RGB -> BGR
        frame = draw_bbox(frame, bbox, color, 2)
        frame = draw_skeleton(frame, kps, color, 1, True)

        # plot attention
        attn_w = attn_w.mean(axis=(1, 0))
        attn_w = attn_w[0::2] + attn_w[1::2]  # sum x and y
        attn_w = np.clip(attn_w, 0.0, vmax)  # (0.0, vmax)
        attn_w = attn_w * (1 / vmax)  # (0.0, 1.0)
        attn_w = (attn_w * 100).astype(int)
        for i in range(len(kps)):
            pt = kps[i].astype(int).tolist()
            w = attn_w[i]
            c = (np.array(_cm_jet(w))[:3] * 255).astype(int).tolist()
            c = c[::-1]
            frame = cv2.circle(frame, pt, 3, c, -1)
        for i in range(len(bbox)):
            pt = bbox[i].astype(int).tolist()
            w = attn_w[i + len(kps)]
            c = (np.array(_cm_jet(w))[:3] * 255).astype(int).tolist()
            c = c[::-1]
            frame = cv2.circle(frame, pt, 3, c, -1)

    return frame


def arange_attention_heatmaps(
    results, n_clusters, n_layers, plot_figsize, vmaxs=(0.5, 0.3, 0.1)
):
    fig = plt.figure(figsize=(plot_figsize[0] / 100, plot_figsize[1] / 100))
    axs = fig.subplots(n_clusters, n_layers)
    for label in range(n_clusters):
        attn_w = np.array([r["attn_w"] for r in results if r["label"] == label])
        if len(attn_w) > 0:
            attn_w = attn_w.mean(axis=0)
            for i in range(n_layers):
                sns.heatmap(
                    attn_w[i],
                    annot=False,
                    cmap="jet",
                    ax=axs[label, i],
                    cbar=False,
                    xticklabels=False,
                    yticklabels=False,
                    vmin=0.0,
                    vmax=vmaxs[i],
                )
        else:
            # set blank
            for i in range(n_layers):
                ax = axs[label, i]
                ax.set_xticks([])
                ax.set_yticks([])
                ax.tick_params(axis="both", color="w")
                ax.spines[["left", "right", "bottom", "top"]].set_visible(False)

    # set titles
    for i in range(n_layers):
        axs[0, i].set_xlabel(f"Layer {i}")
        axs[0, i].xaxis.set_label_position("top")
    for i in range(n_clusters):
        axs[i, 0].set_ylabel(f"Label {i}")

    fig.tight_layout()
    fig.canvas.draw()
    img = np.array(fig.canvas.renderer.buffer_rgba())
    plt.close()
    return img


def plot_attention_clustering_on_frame(
    frame, results, idx_data, frame_size, range_points, vmax=0.5
):
    for data in results:
        label = str(data["label"])
        bbox = data["bbox"][idx_data]
        kps = data["kps"][idx_data]
        attn_w_cls = data["attn_w_cls"]

        bbox = NormalizeBbox.reverse(bbox, frame_size, range_points)
        kps = NormalizeKeypoints.reverse(kps, bbox, range_points)

        # clustering label
        pt = tuple(np.mean(bbox, axis=0).astype(int))
        frame = cv2.putText(
            frame, label, pt, cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2
        )

        # plot bbox and skeleton limbs
        color = (np.array(_cm_tab10(int(label))[:3]) * 255).astype(int).tolist()
        color = tuple(color[::-1])  # RGB -> BGR
        frame = draw_bbox(frame, bbox, color, 2)
        frame = draw_skeleton(frame, kps, color, 1, True)

        # plot attention
        attn_w_cls = attn_w_cls[0, 0, 1:]
        attn_w_cls = attn_w_cls[0::2] + attn_w_cls[1::2]  # sum x and y
        attn_w_cls = np.clip(attn_w_cls, 0.0, vmax)  # (0.0, vmax)
        attn_w_cls = attn_w_cls * (1 / vmax)  # (0.0, 1.0)
        attn_w_cls = (attn_w_cls * 100).astype(int)
        for i in range(len(kps)):
            pt = kps[i].astype(int).tolist()
            w = attn_w_cls[i]
            c = (np.array(_cm_jet(w))[:3] * 255).astype(int).tolist()
            c = c[::-1]
            frame = cv2.circle(frame, pt, 3, c, -1)
        for i in range(len(bbox)):
            pt = bbox[i].astype(int).tolist()
            w = attn_w_cls[i + len(kps)]
            c = (np.array(_cm_jet(w))[:3] * 255).astype(int).tolist()
            c = c[::-1]
            frame = cv2.circle(frame, pt, 3, c, -1)

    return frame


def arange_attention_clustering_heatmaps(results, n_clusters, plot_figsize, vmax=0.5):
    fig = plt.figure(figsize=(plot_figsize[0] / 100, plot_figsize[1] / 100))
    axs = fig.subplots(n_clusters, 1).ravel()
    for label in range(n_clusters):
        attn_w_cls = np.array([r["attn_w_cls"] for r in results if r["label"] == label])
        if len(attn_w_cls) > 0:
            attn_w_cls = attn_w_cls.mean(axis=0)
            sns.heatmap(
                attn_w_cls[0],
                annot=False,
                cmap="jet",
                ax=axs[label],
                cbar=False,
                xticklabels=False,
                yticklabels=False,
                vmin=0.0,
                vmax=vmax,
            )
        else:
            # set blank
            ax = axs[label]
            ax.set_xticks([])
            ax.set_yticks([])
            ax.tick_params(axis="both", color="w")
            ax.spines[["left", "right", "bottom", "top"]].set_visible(False)

    # set titles
    for i in range(n_clusters):
        axs[i].set_ylabel(f"Label {i}")

    fig.tight_layout()
    fig.canvas.draw()
    img = np.array(fig.canvas.renderer.buffer_rgba())
    plt.close()
    return img


def plot_book_idx_on_frame(
    frame, results, idx_data, frame_size, book_size, range_points
):
    cm = plt.get_cmap("turbo", book_size)
    for data in results:
        label = str(data["label"])
        bbox = data["bbox"][idx_data]
        kps = data["kps"][idx_data]
        book_idx = data["book_idx"]

        bbox = NormalizeBbox.reverse(bbox, frame_size, range_points)
        kps = NormalizeKeypoints.reverse(kps, bbox, range_points)

        # clustering label
        pt = tuple(np.mean(bbox, axis=0).astype(int))
        frame = cv2.putText(
            frame, label, pt, cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2
        )

        # plot bbox and skeleton limbs
        color = (np.array(_cm_tab10(int(label))[:3]) * 255).astype(int).tolist()
        color = tuple(color[::-1])  # RGB -> BGR
        frame = draw_bbox(frame, bbox, color, 2)
        frame = draw_skeleton(frame, kps, color, 1, True)

        # plot attention
        for i in range(len(kps)):
            pt = kps[i].astype(int).tolist()
            idx = book_idx[i]
            c = (np.array(cm(idx))[:3] * 255).astype(int).tolist()
            c = c[::-1]
            frame = cv2.circle(frame, pt, 3, c, -1)
        for i in range(len(bbox)):
            pt = bbox[i].astype(int).tolist()
            idx = book_idx[i + len(kps)]
            c = (np.array(cm(idx))[:3] * 255).astype(int).tolist()
            c = c[::-1]
            frame = cv2.circle(frame, pt, 3, c, -1)

    return frame


def arange_book_idx_heatmaps(results, n_clusters, plot_figsize, book_size, vmax=1.0):
    fig = plt.figure(figsize=(plot_figsize[0] / 100, plot_figsize[1] / 100))
    axs = fig.subplots(n_clusters, 1)
    for label in range(n_clusters):
        book_indices = np.array([r["book_idx"] for r in results if r["label"] == label])
        if len(book_indices) > 0:
            book_indices_one_hot = np.eye(book_size)[book_indices]
            book_indices_count = book_indices_one_hot.sum(axis=0)
            book_indices_ratio = book_indices_count / book_indices_count.sum(
                axis=1, keepdims=True
            )
            sns.heatmap(
                book_indices_ratio,
                annot=False,
                cmap="Blues",
                ax=axs[label],
                cbar=False,
                xticklabels=False,
                yticklabels=False,
                vmin=0.0,
                vmax=vmax,
            )
        else:
            # set blank
            ax = axs[label]
            ax.set_xticks([])
            ax.set_yticks([])
            ax.tick_params(axis="both", color="w")
            ax.spines[["left", "right", "bottom", "top"]].set_visible(False)

    # set titles
    for i in range(n_clusters):
        axs[i].set_ylabel(f"Label {i}")

    fig.tight_layout()
    fig.canvas.draw()
    img = np.array(fig.canvas.renderer.buffer_rgba())
    plt.close()
    return img


def moving_average(vals, size):
    b = np.ones(size) / size
    vals_mean = np.convolve(vals, b, mode="same")

    n_conv = math.ceil(size / 2)

    vals_mean[0] *= size / n_conv
    for i in range(1, n_conv):
        vals_mean[i] *= size / (i + n_conv)
        vals_mean[-i] *= size / (i + n_conv - (size % 2))

    return vals_mean


def plot_mse(
    mse_x_dict,
    frame_count,
    stride,
    th,
    ylabel,
    mv_size=10,
    figpath=None,
    is_show=False,
    ylim=(0, 1),
):
    vals_dict = {}
    for _id, mse_dict in mse_x_dict.items():
        if len(mse_dict) < 2:
            continue
        n_frames = np.array(list(mse_dict.keys()))
        mses = np.array(list(mse_dict.values()))

        idxs = n_frames // stride
        n_samples = frame_count // stride + 1
        vals = np.full((n_samples,), np.nan, np.float32)
        vals[idxs] = mses

        vals_dict[_id] = vals.astype(np.float32)

    ids = list(vals_dict.keys())
    vals = np.array(list(vals_dict.values())).T
    vals = vals.astype(np.float32)
    mse_ratio = np.count_nonzero(vals > th, axis=1) / np.count_nonzero(
        np.nan_to_num(vals), axis=1
    )

    # moving average
    mse_ratio = moving_average(mse_ratio, mv_size)

    fig = plt.figure(figsize=(12, 4))
    ax1 = fig.add_subplot(1, 1, 1)
    ax1.plot(vals, color="black", linewidth=1, alpha=0.3, label=ids)
    ax1.set_xlim(0, n_samples)
    ax1.set_xticks(np.arange(0, 1801, 60 * 3), np.arange(0, 1801, 60 * 3) // 60)
    ax1.set_xlabel("Minutes")
    margin = abs(ylim[1] - ylim[0]) * 0.05
    ax1.set_ylim(ylim[0] - margin, ylim[1] + margin)
    ylabels = np.linspace(ylim[0], ylim[1], 6)
    ax1.set_yticks(ylabels)
    ax1.set_ylabel(ylabel)

    ax2 = ax1.twinx()
    ax2.plot(mse_ratio, color="red", linewidth=1, label="mean")
    ax2.set_ylim(-0.05, 1.05)
    ax2.set_ylabel(f"Ratio (MSE > {th})")

    if figpath is not None:
        plt.savefig(figpath, bbox_inches="tight")
    if is_show:
        plt.show()
    plt.close()


def plot_label_ratio_cumsum(
    label_counts,
    classes,
    frame_count,
    stride,
    mv_size=30,
    figpath=None,
    is_show=False,
    ylim_twinx=(0, 15),
):
    cm = plt.get_cmap("tab10")

    vals_dict = {}
    for label, count_dict in label_counts.items():
        if len(count_dict) < 2:
            continue
        n_frames = np.array(list(count_dict.keys()))
        counts = np.array(list(count_dict.values()))

        idxs = n_frames // stride
        n_samples = frame_count // stride + 1
        vals = np.zeros((n_samples,), np.float32)
        vals[idxs] = counts

        vals_dict[label] = vals

    labels = list(vals_dict.keys())
    vals = np.array(list(vals_dict.values())).T

    # moving average
    for i in range(vals.shape[1]):
        vals.T[i] = moving_average(vals.T[i], mv_size)

    # compute ratio
    sums = vals.sum(axis=1, keepdims=True)
    sums[sums == 0] = 1
    vals /= sums

    fig = plt.figure(figsize=(12, 4))
    ax1 = fig.add_subplot(1, 1, 1)
    vals_cumsum = np.cumsum(vals[:, ::-1], axis=1)
    x = np.arange(n_samples)
    for label, val in zip(labels[::-1], vals_cumsum.T):
        c = cm(label)
        if label == 4:
            y1 = np.zeros((n_samples,))
        ax1.fill_between(x, y1, val, facecolor=c, label=classes[label], alpha=0.5)
        y1 = val

    ax1.set_xlim(0, n_samples)
    ax1.set_xlabel("Minutes")
    ax1.set_xticks(np.arange(0, 1801, 60 * 3), np.arange(0, 1801, 60 * 3) // 60)
    ax1.set_ylabel("Label Ratio")
    ax1.set_ylim(0, 1)

    ax2 = ax1.twinx()
    ax2.plot(sums.ravel(), color="black", label="Number of Individuals")
    ax2.set_ylim(ylim_twinx)
    ax2.set_ylabel("Number of Individuals")

    ax1_handles, ax1_labels = ax1.get_legend_handles_labels()
    ax2_handles, ax2_labels = ax2.get_legend_handles_labels()
    handles = ax1_handles[::-1] + ax2_handles
    labels = ax1_labels[::-1] + ax2_labels
    ax1.legend(handles, labels, bbox_to_anchor=(1.05, 1), loc="upper left")

    if figpath is not None:
        plt.savefig(figpath, bbox_inches="tight")
    if is_show:
        plt.show()
    plt.close()


def plot_tsne(
    X,
    labels,
    classes,
    perplexity=10,
    figpath=None,
    is_show=False,
    cmap="tab10",
    lut=None,
    legend=True,
):
    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity, max_iter=1000)
    embedded = tsne.fit_transform(X)
    unique_labels = np.unique(labels)
    cm = plt.get_cmap(cmap, lut)
    plt.figure(figsize=(5, 5))
    for label in unique_labels:
        x = embedded[labels == label]
        if lut is not None:
            ci = np.where(unique_labels == label)[0].item()
        else:
            ci = int(label)
        c = cm(ci)
        plt.scatter(x.T[0], x.T[1], s=3, c=c, label=classes[label])
    if legend:
        plt.legend(bbox_to_anchor=(1.01, 1), loc="upper left")
    if figpath is not None:
        plt.savefig(figpath, bbox_inches="tight")
    if is_show:
        plt.show()
    plt.close()


def plot_cm(cm, labels, figpath=None, normalize=False, on_plot=True):
    array = cm / (
        (cm.sum(0).reshape(1, -1) + 1e-9) if normalize else 1
    )  # normalize columns
    array[array < 0.005] = np.nan  # don't annotate (would appear as 0.00)

    fig, ax = plt.subplots(1, 1, figsize=(8, 6), tight_layout=True)
    ticklabels = labels
    vmax = 1.0 if normalize else None
    sns.heatmap(
        array,
        ax=ax,
        annot=True,
        annot_kws={"size": 8},
        cmap="Blues",
        fmt=".2f" if normalize else ".0f",
        square=True,
        vmin=0.0,
        vmax=vmax,
        xticklabels=ticklabels,
        yticklabels=ticklabels,
    ).set_facecolor((1, 1, 1))

    ax.set_xlabel("True")
    ax.set_xticklabels(labels, rotation=90)
    ax.set_ylabel("Predicted")
    ax.set_yticklabels(labels, rotation=0)
    if figpath is not None:
        if normalize:
            figpath = figpath.replace(".png", "") + "_normalized.png"
        fig.savefig(figpath, bbox_inches="tight")
    if on_plot:
        plt.show()
    plt.close(fig)
