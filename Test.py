import unittest
from pipeline import (
    normalize_answer,
    squad_f1,
    squad_recall,
    calculate_core_token_preservation,
    extract_agent_understanding,
    extract_content_tokens,
    squad_em,
    select_checkpoints,
    measure_f1_recoverability_loss,
    calculate_semantic_drift
)


class TestNormalizeAnswer(unittest.TestCase):
    def test_lowercases_and_strips_punctuation(self):
        self.assertEqual(normalize_answer("The University, of Chicago."),
                         "university of chicago")

    def test_removes_articles(self):
        self.assertEqual(normalize_answer("a cat and an apple the end"),
                         "cat and apple end")

    def test_collapses_whitespace(self):
        self.assertEqual(normalize_answer("  hello    world  "), "hello world")

    def test_empty_string(self):
        self.assertEqual(normalize_answer(""), "")


class TestSquadF1(unittest.TestCase):
    def test_exact_match_scores_one(self):
        self.assertAlmostEqual(squad_f1("cnidarians", "cnidarians"), 1.0)

    def test_no_overlap_scores_zero(self):
        self.assertEqual(squad_f1("cnidaria", "cnidarians"), 0.0)

    def test_verbose_answer_penalised_by_precision(self):
        f1 = squad_f1("the tower theatre was built in 1939", "1939")
        self.assertAlmostEqual(f1, 2 * (1/6) * 1.0 / ((1/6) + 1.0), places=4)

    def test_partial_overlap(self):
        self.assertAlmostEqual(squad_f1("University of Chicago", "Chicago University"), 0.8, places=4)


class TestSquadRecall(unittest.TestCase):
    def test_verbose_answer_full_recall(self):
        self.assertEqual(squad_recall("the tower theatre was built in 1939", "1939"), 1.0)

    def test_missing_token_zero_recall(self):
        self.assertEqual(squad_recall("phylum cnidaria", "cnidarians"), 0.0)

    def test_partial_recall(self):
        self.assertEqual(squad_recall("betty", "betty meggers"), 0.5)

    def test_empty_ground_truth(self):
        self.assertEqual(squad_recall("anything", ""), 0.0)


class TestCoreTokenPreservation(unittest.TestCase):
    def test_identical_text_full_preservation(self):
        score = calculate_core_token_preservation(
            "The rainforest contains diverse species.",
            "The rainforest contains diverse species.",
        )
        self.assertAlmostEqual(score, 1.0, places=4)

    def test_bounds(self):
        score = calculate_core_token_preservation(
            "Betty Meggers proposed the theory.",
            "Something completely different entirely.",
        )
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)


class TestExtractContentTokens(unittest.TestCase):
    def test_keeps_content_words_drops_stopwords(self):
        tokens = extract_content_tokens("The rainforest contains diverse species")
        self.assertIn("rainforest", tokens)
        self.assertIn("specie", tokens)     
        self.assertNotIn("the", tokens)

    def test_returns_a_set(self):
        self.assertIsInstance(extract_content_tokens("Betty Meggers proposed a theory"), set)

    def test_empty_string(self):
        self.assertEqual(extract_content_tokens(""), set())


class TestSquadEM(unittest.TestCase):
    def test_exact_match_after_normalisation(self):
        self.assertEqual(squad_em("The University, of Chicago.", "university of chicago"), 1.0)

    def test_mismatch_scores_zero(self):
        self.assertEqual(squad_em("Paris", "London"), 0.0)


class TestSelectCheckpoints(unittest.TestCase):
    def test_depth_10(self):
        self.assertEqual(select_checkpoints(10), [1, 5, 10])

    def test_depth_5(self):
        self.assertEqual(select_checkpoints(5), [1, 3, 5])

    def test_first_and_last_always_present(self):
        cps = select_checkpoints(7)
        self.assertIn(1, cps)
        self.assertIn(7, cps)


class TestRecoverabilityLossOnImpossible(unittest.TestCase):
    def test_refusal_on_impossible_is_zero_loss(self):
        understanding = {"answer": "CANNOT_ANSWER"}
        loss = measure_f1_recoverability_loss(understanding, [], is_impossible=True)
        self.assertEqual(loss, 0.0)

    def test_fabrication_on_impossible_is_full_loss(self):
        understanding = {"answer": "Some invented answer"}
        loss = measure_f1_recoverability_loss(understanding, [], is_impossible=True)
        self.assertEqual(loss, 1.0)


class TestSemanticDriftContract(unittest.TestCase):
    def test_identical_text_zero_drift(self):
        d = calculate_semantic_drift("The rainforest is dense.", "The rainforest is dense.")
        self.assertAlmostEqual(d, 0.0, places=3)

    def test_drift_in_valid_range(self):
        d = calculate_semantic_drift("Cats are mammals.", "Quantum physics is complex.")
        self.assertGreaterEqual(d, 0.0)
        self.assertLessEqual(d, 1.0)

if __name__ == "__main__":
    unittest.main(verbosity=2)