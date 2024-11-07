from utils import generate_all_binary_vectors


def test_all_possible_concept_vectors():
    """test the functionality of all_cs"""
    all_cs = generate_all_binary_vectors(length=3)
    assert all_cs == [[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1], [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]]