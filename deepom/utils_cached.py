from pathlib import Path

import fire

from deepom.utils import extract_segment_from_endpoints


def read_jxr(file):
    return imagecodecs.jpegxr_decode(Path(file).read_bytes())


def read_jxr_segment(file, endpoints, segment_width):
    return extract_segment_from_endpoints(read_jxr(file)[None], endpoints=endpoints,
                                          segment_width=segment_width)


if __name__ == '__main__':
    fire.Fire()
