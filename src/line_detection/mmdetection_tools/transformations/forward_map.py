import os

os.sys.path.append("/workspaces/doc-matcher/src")

import numpy as np
import matplotlib.pyplot as plt

from typing import Optional
from copy import deepcopy
from pathlib import Path
from matplotlib import gridspec
from matplotlib.cm import ScalarMappable
from shapely.geometry import LineString
from inv3d_util.load import load_image
from inv3d_util.mapping import invert_map, apply_map, scale_map

from .forward_map_channel import ForwardMapChannel
from .util import NoStdStreams


class ForwardMap:

    def __init__(
        self,
        warped_image_file: Optional[str],
        min_slope: float = 0,
        max_slope: float = 0.0025,
    ):
        if warped_image_file is not None:
            self.warped_image_file = Path(warped_image_file)
            self.warped_image = load_image(self.warped_image_file)
        else:
            self.warped_image_file = None
            self.warped_image = None

        top_border = LineString([(0, 0), (512, 0)])
        bottom_border = LineString([(0, 512), (512, 512)])
        left_border = LineString([(0, 0), (0, 512)])
        right_border = LineString([(512, 0), (512, 512)])

        vertical_borders = [top_border, bottom_border]
        horizontal_borders = [left_border, right_border]

        self.v_map = ForwardMapChannel(
            directional_borders=vertical_borders,
            orthogonal_borders=horizontal_borders,
            min_slope=min_slope,
            max_slope=max_slope,
            is_vertical_gradient=True,
        )

        self.h_map = ForwardMapChannel(
            directional_borders=horizontal_borders,
            orthogonal_borders=vertical_borders,
            min_slope=min_slope,
            max_slope=max_slope,
            is_vertical_gradient=False,
        )

    def add_line(self, warped_line: LineString, template_line: LineString, type: str):
        old_v_map = deepcopy(self.v_map)
        old_h_map = deepcopy(self.h_map)

        valid_v = self.v_map.add_line(warped_line, template_line, type)
        valid_h = self.h_map.add_line(warped_line, template_line, type)

        if not valid_v or not valid_h:
            # print("Reverting both changes.")
            self.v_map.restore(old_v_map)
            self.h_map.restore(old_h_map)

    def add_horizontal_line(
        self, warped_line: LineString, template_line: LineString, type: str
    ):
        self.v_map.add_line(warped_line, template_line, type)

    def add_vertical_line(
        self, warped_line: LineString, template_line: LineString, type: str
    ):
        self.h_map.add_line(warped_line, template_line, type)

    def project_points(self):
        self.v_map.project_points()
        self.h_map.project_points()

    def create_uv_map(self) -> np.ndarray:
        return np.stack([self.v_map.interp, self.h_map.interp], axis=-1)

    def create_fast_bm_map(self) -> np.ndarray:
        return invert_map(self.create_uv_map())

    def create_good_bm_map(self) -> np.ndarray:
        uv = self.create_uv_map()

        with NoStdStreams():
            uv_intermediate = scale_map(uv, resolution=1024)
        bm_intermediate = invert_map(uv_intermediate)

        bm = scale_map(bm_intermediate, 512)
        return bm

    def visualize(self, output_file: Optional[Path], show_convex_hull: bool = False):
        v_points = np.array(self.v_map.points)
        h_points = np.array(self.h_map.points)

        color_map = {
            "start": "blue",
            "matched": "green",
            "projected": "orange",
            "unmatched": "purple",
        }

        bm = self.create_fast_bm_map()

        unwarped = apply_map(self.warped_image, bm, resolution=(512, 512))

        plt.clf()
        plt.figure(figsize=(15, 15))

        # Create a 2x2 grid with specified widths for the subplots and color bar
        gs = gridspec.GridSpec(2, 3, width_ratios=[1, 1, 0.05])

        # Subplot 1
        ax1 = plt.subplot(gs[0, 0])
        ax1.imshow(self.warped_image)
        ax1.imshow(self.v_map.interp, alpha=0.5)
        ax1.triplot(v_points[:, 0], v_points[:, 1], self.v_map.tri.simplices)
        for i, point_type in enumerate(self.v_map.types):
            color = color_map.get(point_type, "black")
            ax1.plot(v_points[i, 0], v_points[i, 1], "o", color=color)
        ax1.set_title("Vertical interpolation")
        ax1.set_xlim(-2, 514)  # Set x limits
        ax1.set_ylim(-2, 514)  # Set y limits
        ax1.invert_yaxis()

        if show_convex_hull:
            hull = self.v_map.convex_hull("matched")
            if hull is not None:
                ax1.plot(*hull.exterior.xy, color="orange")

        # Subplot 2
        ax2 = plt.subplot(gs[0, 1])
        ax2.imshow(self.warped_image)
        ax2.imshow(self.h_map.interp, alpha=0.5)
        ax2.triplot(h_points[:, 0], h_points[:, 1], self.h_map.tri.simplices)
        for i, point_type in enumerate(self.h_map.types):
            color = color_map.get(point_type, "black")
            ax2.plot(h_points[i, 0], h_points[i, 1], "o", color=color)
        ax2.set_title("Horizontal interpolation")
        ax2.set_xlim(-2, 514)  # Set x limits
        ax2.set_ylim(-2, 514)  # Set y limits
        ax2.invert_yaxis()

        if show_convex_hull:
            hull = self.h_map.convex_hull("matched")
            if hull is not None:
                ax2.plot(*hull.exterior.xy, color="orange")

        # Subplot 3
        ax3 = plt.subplot(gs[1, 0])
        ax3.imshow(self.warped_image)
        ax3.set_title("Warped image")

        # Subplot 4
        ax4 = plt.subplot(gs[1, 1])
        ax4.imshow(unwarped)
        ax4.set_title("Unwarped image")

        # Adjust layout to prevent overlap
        plt.tight_layout()

        # Add color bar to the right of Subplot 2
        sm = ScalarMappable(cmap=plt.cm.viridis, norm=plt.Normalize(vmin=0, vmax=1))
        sm.set_array([])  # You need to set an array for the ScalarMappable

        # Add color bar to the right of Subplot 2
        cax = plt.subplot(gs[0, 2])
        cbar = plt.colorbar(sm, cax=cax)
        cbar.ax.invert_yaxis()

        # Show the plot
        if output_file is None:
            plt.show()
        else:
            plt.savefig(output_file.as_posix())
