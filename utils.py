import numpy as np
import matplotlib.pyplot as plt
import cv2

_OFF_WHITE = (1.0, 1.0, 240.0 / 255)
_LARGE_MASK_AREA_THRESH = 120000


def draw_text(
    ax,
    text,
    position,
    *,
    font_size=None,
    color="g",
    horizontal_alignment="center",
    rotation=0
):
    """
    Args:
        text (str): class label
        position (tuple): a tuple of the x and y coordinates to place text on image.
        font_size (int, optional): font of the text. If not provided, a font size
            proportional to the image width is calculated and used.
        color: color of the text. Refer to `matplotlib.colors` for full list
            of formats that are accepted.
        horizontal_alignment (str): see `matplotlib.text.Text`
        rotation: rotation angle in degrees CCW

    Returns:
        output (VisImage): image object with text drawn.
    """
    if not font_size:
        font_size = 10

    # since the text background is dark, we don't want the text to be dark
    # color = np.maximum(list(mplc.to_rgb(color)), 0.2)
    # color[np.argmax(color)] = max(0.8, np.max(color))

    x, y = position
    ax.text(
        x,
        y,
        text,
        size=font_size,
        family="sans-serif",
        bbox={"facecolor": "black", "alpha": 0.8, "pad": 0.7, "edgecolor": "none"},
        verticalalignment="top",
        horizontalalignment=horizontal_alignment,
        color=color,
        zorder=10,
        rotation=rotation,
    )
    # return self.output


def draw_binary_mask(ax, binary_mask, color=None, *, edge_color=None, text=None, alpha=0.5):
    """
    Args:
        binary_mask (ndarray): numpy array of shape (H, W), where H is the image height and
            W is the image width. Each value in the array is either a 0 or 1 value of uint8
            type.
        color: color of the mask. Refer to `matplotlib.colors` for a full list of
            formats that are accepted. If None, will pick a random color.
        edge_color: color of the polygon edges. Refer to `matplotlib.colors` for a
            full list of formats that are accepted.
        text (str): if None, will be drawn in the object's center of mass.
        alpha (float): blending efficient. Smaller values lead to more transparent masks.
        area_threshold (float): a connected component small than this will not be shown.

    Returns:
        output (VisImage): image object with mask drawn.
    """

    # if color is None:
    #     color = random_color(rgb=True, maximum=1)
    # color = mplc.to_rgb(color)

    binary_mask = binary_mask.astype("uint8")  # opencv needs uint8
    shape2d = (binary_mask.shape[0], binary_mask.shape[1])

    rgba = np.zeros(shape2d + (4,), dtype="float32")
    rgba[:, :, :3] = color
    rgba[:, :, 3] = (binary_mask == 1).astype("float32") * alpha
    ax.imshow(rgba)  # , extent=(0, , self.output.height, 0))

    if text is not None:
        # TODO sometimes drawn on wrong objects. the heuristics here can improve.
        # lighter_color = self._change_color_brightness(color, brightness_factor=0.7)
        lighter_color = (1, 1, 1)
        _num_cc, cc_labels, stats, centroids = cv2.connectedComponentsWithStats(binary_mask, 8)
        largest_component_id = np.argmax(stats[1:, -1]) + 1

        # draw text on the largest component, as well as other very large components.
        for cid in range(1, _num_cc):
            if cid == largest_component_id or stats[cid, -1] > _LARGE_MASK_AREA_THRESH:
                # median is more stable than centroid
                center = np.median((cc_labels == cid).nonzero(), axis=1)[::-1]
                draw_text(ax, text, center, color=lighter_color)
