# -*- coding: utf-8 -*-
import ccm
import numpy as np
from scipy import sparse
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import unittest


class TestCCM(unittest.TestCase):
    def setUp(self):
        node_table = pd.read_csv("tests/data/cora-node_table.csv")
        edge_table = pd.read_csv(
            "tests/data/cora-edge_table.csv",
            dtype={"src": np.int32, "trg": np.int32},
        )
        src, trg = tuple(edge_table[["src", "trg"]].values.T)

        rows, cols = src, trg
        nrows, ncols = node_table.shape[0], node_table.shape[0]
        A = sparse.csr_matrix(
            (np.ones_like(rows), (rows, cols)),
            shape=(nrows, ncols),
        ).asfptype()

        self.A = A
        self.node_labels = np.unique(node_table["field"].values, return_inverse=True)[1]

    def test_fit(self):

        model = ccm.CCM(dim=32, c0=5, reg_kappa=1e-5)
        model.fit(self.A, epochs=30)

        emb = model.get_embedding()
        eta = model.get_fitness()
        kappa = model.get_kappa()
        lam = model.get_lambda()

        self.assertEqual(emb.shape, (self.A.shape[0], 32))
        self.assertEqual(eta.shape, (self.A.shape[0],))

        # Evaluate embedding
        clf = LinearDiscriminantAnalysis(
            n_components=len(np.unique(self.node_labels)) - 1
        )
        clf.fit(emb, self.node_labels)
        score = clf.score(emb, self.node_labels)
        self.assertGreater(score, 0.7)

    def test_forecast(self):
        model = ccm.CCM(dim=32, c0=5, reg_kappa=1e-5)
        model.fit(self.A, epochs=30)

        n_refs = 30
        t_publish = [np.ones(50) * i for i in range(30)]
        n_refs = np.ones_like(t_publish) * n_refs

        A_gen = model.forecast(t_publish=t_publish, n_refs=n_refs)
        self.assertEqual(A_gen.shape[0], self.A.shape[0] + len(t_publish))
        self.assertEqual(A_gen.sum() - self.A.sum(), n_refs.sum())

    def test_regenerate(self):
        model = ccm.CCM(dim=32, c0=5, reg_kappa=1e-5)
        model.fit(self.A, epochs=30)

        A_gen = model.regenerate()
        self.assertEqual(A_gen.shape, self.A.shape)
        self.assertEqual(A_gen.sum() - self.A.sum(), 0)


if __name__ == "__main__":
    unittest.main()
