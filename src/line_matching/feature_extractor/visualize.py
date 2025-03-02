import cv2
from typing import Dict, List, Optional, Any
import matplotlib.pyplot as plt
from shapely.geometry import LineString, Polygon
from positional_encodings.torch_encodings import (
    PositionalEncoding2D,
)
import numpy as np
import torch

from skimage.transform import resize
from skimage import img_as_bool


def visualize_visual_embedding(vis_embedding: np.ndarray) -> plt.Figure:
    plt.clf()
    return plt.imshow(vis_embedding)


def visualize_cartesian_positional_embedding(
    resolution: int, line: LineString, margin: float, depth: int
) -> plt.Figure:
    plt.clf()
    pos_encoding = PositionalEncoding2D(depth)(
        torch.zeros(1, resolution, resolution, depth)
    ).squeeze(0)

    polygon = line.buffer(margin)
    mask = polygon2mask(polygon=polygon, resolution=512, rescale_size=None)
    mask = mask.transpose(1, 0)

    return visualize_positional_embedding(pos_embedding=pos_encoding, mask=mask)


def visualize_positional_embedding(
    pos_embedding: np.ndarray, mask: Optional[np.ndarray] = None
) -> plt.Figure:
    plt.clf()
    height = pos_embedding.shape[0]
    width = pos_embedding.shape[1]
    channels = pos_embedding.shape[-1]

    assert channels % 4 == 0, "Channels must be divisible by 4"

    row_length = int(channels / 2)

    fig, axs = plt.subplots(ncols=row_length, nrows=2)

    vmin = -1
    vmax = 1

    def prepare_axis(ax, data, show_x, show_y):
        mask_hide = np.ma.masked_where(mask != 0, data)
        mask_show = np.ma.masked_where(mask == 0, data)

        ax.imshow(
            mask_hide,
            cmap="GnBu",
            interpolation="none",
            vmin=vmin,
            vmax=vmax,
            alpha=0.5,
        )
        ax.imshow(
            mask_show,
            cmap="GnBu",
            interpolation="none",
            vmin=vmin,
            vmax=vmax,
        )

        if not show_x:
            ax.tick_params(
                axis="x",
                which="both",
                bottom=False,
                top=False,
                labelbottom=False,
            )

        if not show_y:
            ax.tick_params(
                axis="y",
                which="both",
                left=False,
                right=False,
                labelleft=False,
            )

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.spines["left"].set_visible(False)

        ax.axis("equal")

    for x in range(row_length):
        for y in range(2):
            prepare_axis(
                ax=axs[y, x],
                data=pos_embedding[..., x + y * row_length],
                show_x=y == 1,
                show_y=x == 0,
            )

    plt.subplots_adjust(wspace=0.05, hspace=0.05)

    fig_width = 16
    fig_height = (fig_width * height * 2) / (width * row_length)

    fig.set_size_inches(fig_width, fig_height)

    return fig


def visualize_full_embedding(
    vis_embedding: np.ndarray, pos_embedding: np.ndarray, title: Optional[str] = None
) -> plt.Figure:
    plt.clf()
    pos_channels = int(pos_embedding.shape[2])

    fig, axs = plt.subplots(ncols=1, nrows=pos_channels + 1)

    vmin = -1
    vmax = 1

    def prepare_axis(ax, data, show_x, show_y, index):
        ax.imshow(
            data,
            cmap="GnBu",
            interpolation="none",
            vmin=vmin,
            vmax=vmax,
        )

        if not show_x:
            ax.tick_params(
                axis="x",
                which="both",
                bottom=False,
                top=False,
                labelbottom=False,
            )

        if not show_y:
            ax.tick_params(
                axis="y",
                which="both",
                left=False,
                right=False,
                labelleft=False,
            )

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.spines["left"].set_visible(False)

        ax.axis("equal")

        ax.set_ylabel(f"positional {index}", rotation=0, ha="right", va="center")

    axs[0].imshow(vis_embedding)
    axs[0].tick_params(
        axis="x",
        which="both",
        bottom=False,
        top=False,
        labelbottom=False,
    )
    axs[0].tick_params(
        axis="y",
        which="both",
        left=False,
        right=False,
        labelleft=False,
    )
    axs[0].set_ylabel("visual", rotation=0, ha="right", va="center")

    for p in range(pos_channels):
        prepare_axis(
            ax=axs[p + 1],
            data=pos_embedding[..., p],
            show_x=p == pos_channels - 1,
            show_y=False,
            index=p,
        )

    plt.subplots_adjust(wspace=0.05, hspace=0.05)

    fig.set_size_inches(16, 6)

    if title:
        fig.suptitle(title, y=0.92)

    return fig


def visualize_multiple_full_embeddings(
    embeddings: List[Dict[str, Any]], output_file: str
):

    renderings = []

    for data in embeddings:
        fig = visualize_full_embedding(
            data["visual"],
            data["positional"],
            data["title"],
        )
        fig.canvas.draw()
        img_array = np.array(fig.canvas.renderer.buffer_rgba())
        renderings.append(img_array)

    full_rendering = np.concatenate(renderings, axis=0)
    full_rendering = cv2.cvtColor(full_rendering, cv2.COLOR_RGB2BGR)

    cv2.imwrite(output_file, full_rendering)


def polygon2mask(
    polygon: Polygon, resolution: int, rescale_size: Optional[int]
) -> np.ndarray:
    height = resolution
    width = resolution

    mask = np.zeros([height, width])
    points = [[y, x] for x, y in zip(*polygon.boundary.coords.xy)]

    mask = cv2.fillPoly(mask, np.array([points]).astype(np.int32), color=1)
    mask = mask.astype(bool)

    if rescale_size:
        resized = img_as_bool(resize(mask, (rescale_size, rescale_size)))
        resized = np.expand_dims(resized, axis=0)
    else:
        resized = mask

    return resized  # shape: (1, rescale_size, rescale_size)
