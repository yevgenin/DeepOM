import numpy
from numba import njit
from numpy import ndarray

from deepom.utils import is_sorted, ndargmax


class Aligner:
    score_matrix: ndarray = None
    prev_matrix: ndarray = None

    score: float = None
    path_stop: tuple[float, float] = None
    path: ndarray = None
    align_params: dict = {}
    image_len = 0
    
    def add_offset(self,offset):
        self.offset = offset

    def make_alignment(self, qry, ref):
        assert is_sorted(qry) and is_sorted(ref)
        self.qry = qry
        self.ref = ref
        self.compute_score_matrix()
        self.traverse_alignment_path()
        r, q = self.path.T
        self.alignment_ref = self.ref[r]
        self.alignment_qry = self.qry[q]

    def traverse_alignment_path(self):
        r0, q0 = r, q = ndargmax(self.score_matrix)
        s = self.score_matrix[r, q]

        path = []
        while r >= 0 and q >= 0:
            path.append((r, q))
            r, q = self.prev_matrix[r, q]
        path = numpy.stack(path)[::-1]

        self.path = path
        self.path_stop = r0, q0
        self.score = s

    def compute_score_matrix(self):
        self.score_matrix, self.prev_matrix = self._compute_score_matrix(
            qvec=self.qry, rvec=self.ref,
            **self.align_params
        )

    @staticmethod
    @njit(nogil=True)
    def _compute_score_matrix(
            qvec: ndarray = None,
            rvec: ndarray = None,
            loc_factor: float = 500,
            skip_r_factor: float = 10,
            skip_q_factor: float = 10,
            dp_band_size: int = 5,
    ):
        assert 2 <= len(rvec) <= 10 ** 9
        assert 2 <= len(qvec) <= 1000

        len_r = len(rvec)
        len_q = len(qvec)

        S = numpy.full((len_r, len_q), numpy.nan)
        P = numpy.full((len_r, len_q, 2), -1)
        S[0, :] = 0
        S[:, 0] = 0

        for r in range(1, len_r):
            for q in range(1, len_q):
                best_score = -numpy.inf
                best_prev_node = None

                for q0 in range(max(0, q - dp_band_size), q):
                    delta_query = qvec[q] - qvec[q0]
                    skip_query = q - q0 - 1

                    for r0 in range(max(0, r - dp_band_size), r):
                        delta_ref = rvec[r] - rvec[r0]
                        skip_ref = r - r0 - 1

                        delta_cost = abs(delta_ref - delta_query)
                        edge_cost = (
                                delta_cost / loc_factor +
                                skip_ref / skip_r_factor +
                                skip_query / skip_q_factor
                        )
                        node_score = 1
                        cand_score = S[r0, q0] + node_score - edge_cost

                        if cand_score > best_score:
                            best_score = cand_score
                            best_prev_node = r0, q0

                S[r, q] = best_score
                P[r, q, :] = best_prev_node

        return S, P
