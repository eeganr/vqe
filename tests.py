import quantumsimulator as qs
import unittest
import numpy as bknd

class TestSimulator(unittest.TestCase):
    def test_comp_basis(self):
        psi = bknd.ones((2, 2, 2)) / bknd.sqrt(8)

        zero = bknd.array([1, 0])
        plus = bknd.array([1, 1]) / bknd.sqrt(2)
        right = bknd.array([1, 1j]) / bknd.sqrt(2)

        psi = bknd.kron(bknd.kron(zero, plus), right).reshape(2, 2, 2)

        psi = bknd.ones(3) / bknd.sqrt(3)
        
        probs = qs.measure_single_site(psi, 0, check_valid=True)[0]

        # psi = bknd.array([[1, 0], [0, 1]])/bknd.sqrt(2)

        # print(measure_single_site(psi, 0))
        # print(measure_remove_single_site(psi, 2, check_valid=True)) #, basis=bknd.array([[1, 1], [1j, -1j]])/bknd.sqrt(2)))

        bknd.testing.assert_almost_equal(probs, [1/3, 1/3, 1/3])

        bknd.testing.assert_almost_equal(sum(probs), 1)

    def test_alt_basis(self):
        psi = bknd.ones((2, 2, 2)) / bknd.sqrt(8)

        zero = bknd.array([1, 0])
        plus = bknd.array([1, 1]) / bknd.sqrt(2)
        right = bknd.array([1, 1j]) / bknd.sqrt(2)

        psi = bknd.kron(bknd.kron(zero, plus), right).reshape(2, 2, 2)

        psi = bknd.ones(3) / bknd.sqrt(3)
        
        probs = qs.measure_single_site(psi, 0, check_valid=True, basis=bknd.array([[[1, 0, 0], [0, 1, 0], [0, 0, 0]],
                                                                    # [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
                                                                    [[0, 0, 0], [0, 0, 0], [0, 0, 1]]]))[0]

        # psi = bknd.array([[1, 0], [0, 1]])/bknd.sqrt(2)

        # print(measure_single_site(psi, 0))
        # print(measure_remove_single_site(psi, 2, check_valid=True)) #, basis=bknd.array([[1, 1], [1j, -1j]])/bknd.sqrt(2)))

        bknd.testing.assert_almost_equal(probs, [2/3, 1/3], 5)
        
        self.assertAlmostEqual(sum(probs), 1)
        
    def test_alt_basis2(self):
        psi = bknd.ones((2, 2, 2)) / bknd.sqrt(8)

        zero = bknd.array([1, 0])
        plus = bknd.array([1, 1]) / bknd.sqrt(2)
        right = bknd.array([1, 1j]) / bknd.sqrt(2)

        psi = bknd.kron(bknd.kron(zero, plus), right).reshape(2, 2, 2)

        psi = bknd.ones(3) / bknd.sqrt(3)
        
        probs = qs.measure_single_site(psi, 0, check_valid=True, basis=bknd.array([[[1, 0, 0], [0, 1, 0], [0, 0, 0]],
                                                                    # [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
                                                                    [[0, 0, 0], [0, 0, 0], [0, 0, 1]]]))[0]


        bknd.testing.assert_almost_equal(probs, [2/3, 1/3], 5)
        
        self.assertAlmostEqual(sum(probs), 1)


if __name__ == '__main__':
    unittest.main()
    
