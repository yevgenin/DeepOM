import numpy as np
from scipy.interpolate import interp1d

from deepom.aligner import DeepOMAligner
from utils.pyutils import NDArray


class AlignmentAnalysis(DeepOMAligner.Output):
    references: dict

    def __repr__(self):
        return f"{self.reference_id} {self.orientation.name} " \
               f"ref_lims={[*self.ref_lims]} " \
               f"query_lims={[*self.query_lims]} " \
               f"score={self.score:.2f} "

    def reference_segment(self, start=None, stop=None):
        reference = self.reference
        if start is None and stop is None:
            return reference[slice(*self.reference_indices[[0, -1]])]
        else:
            return reference[slice(*reference.searchsorted([start, stop]))]

    @property
    def ref_lims(self):
        return self.aligned_reference[[0, -1]]

    @property
    def query_lims(self):
        return sorted(self.aligned_query[[0, -1]])

    @property
    def reference(self):
        return self.references[self.reference_id]

    def transform_to_reference(self, query_positions: NDArray | list):
        return interp1d(self.aligned_query, self.aligned_reference, bounds_error=False,
                        fill_value="extrapolate")(np.asarray(query_positions))

    def transform_to_query(self, reference_positions: NDArray | list):
        return interp1d(self.aligned_reference, self.aligned_query, bounds_error=False,
                        fill_value="extrapolate")(np.asarray(reference_positions))

    @property
    def aligned_reference(self):
        return self.references[self.reference_id][self.reference_indices]

    @property
    def aligned_query(self):
        return self.query_locs[self.query_indices]
