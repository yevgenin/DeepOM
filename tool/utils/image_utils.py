import numpy as np
from skimage.transform import EuclideanTransform, warp


def extract_segment_from_endpoints(image: np.ndarray, endpoints: np.ndarray, segment_width: float):
    """
    Extract a segment from an image given endpoints of the segment

    :param image: 3D image to extract segment from, dimensions are (channels, height, width)
    :param endpoints: endpoints of the segment: [[y0, x0], [y1, x1]]
    :param segment_width: width of the segment
    """
    (y0, x0), (y1, x1) = endpoints

    segment_angle = np.arctan2(y1 - y0, x1 - x0)
    segment_length = np.sqrt((y0 - y1) ** 2 + (x0 - x1) ** 2)
    segment_center = np.stack([x0 + x1, y0 + y1]) / 2

    T = EuclideanTransform(translation=-segment_center)
    R = EuclideanTransform(rotation=-segment_angle)
    T2 = EuclideanTransform(translation=[segment_length / 2, segment_width / 2])
    M = EuclideanTransform(matrix=T2.params @ R.params @ T.params)

    segment_image = warp(
        image.astype(float).transpose(1, 2, 0),
        inverse_map=M.inverse,
        output_shape=(int(segment_width), int(segment_length))
    ).transpose(2, 0, 1)
    return segment_image
