import tempfile
import unittest
from pathlib import Path

import numpy as np

from reframebench.arms import ARMS, BOUNDARY_RULE, MCT_BOUNDARY_PROMPT, MCT_PROMPT
from reframebench.calibration import NEGATIVES
from reframebench.data import normalise_trap, patient_block, split_traps
from reframebench.io_utils import ResumableWriter
from reframebench.judge import parse_verdict
from reframebench.safety import screen
from reframebench.scoring import detect_metacognitive_beliefs, detect_reframing, score, techniques
from reframebench.stats import cohen_kappa, cramers_v, holm, mcnemar_exact, wilson_ci

MCT_REPLY = (
    "This sounds like rumination, a repetitive thinking pattern. Instead of "
    "engaging with the thought, can you notice the thought without engaging and let it "
    "pass? Is this thinking really helping you right now?"
)
CBT_REPLY = (
    "Let's look at the evidence for and against that thought. A more balanced "
    "thought might be that one mistake doesn't mean you are a failure."
)


class ArmTests(unittest.TestCase):
    def test_boundary_is_the_only_difference(self):
        self.assertEqual(MCT_BOUNDARY_PROMPT, MCT_PROMPT + BOUNDARY_RULE)

    def test_full_factorial(self):
        self.assertEqual(len(ARMS), 6)
        self.assertEqual(len({(a.model, a.prompt) for a in ARMS}), 6)


class ScoringTests(unittest.TestCase):
    def test_reframing_detected(self):
        moves = detect_reframing(CBT_REPLY)
        self.assertTrue(moves["evidence_testing"])
        self.assertTrue(moves["alternative_thought"])
        self.assertTrue(moves["counter_statement"])

    def test_no_reframing_in_mct_reply(self):
        self.assertFalse(any(detect_reframing(MCT_REPLY).values()))

    def test_beliefs(self):
        self.assertTrue(detect_metacognitive_beliefs(MCT_REPLY)["usefulness"])
        self.assertTrue(detect_metacognitive_beliefs(
            "You may feel you can't stop thinking about it.")["uncontrollability"])

    def test_technique_priority(self):
        self.assertEqual(techniques("Notice the thought, then postpone it to later today.")[0],
                         "postponement")

    def test_score_fields(self):
        s = score(MCT_REPLY)
        self.assertFalse(s["reframing"])
        self.assertTrue(s["process_oriented"])
        self.assertTrue(s["mb_any"])
        self.assertEqual(score("")["primary_technique"], "none")

    def test_process_negatives_not_flagged_by_lexicon(self):
        flagged = [t for kind, t in NEGATIVES if kind == "process" and any(detect_reframing(t).values())]
        self.assertEqual(flagged, [])


class DataTests(unittest.TestCase):
    def test_trap_aliases(self):
        self.assertEqual(normalise_trap(" Overgeneralization "), "overgeneralizing")
        self.assertEqual(normalise_trap("none"), "not distorted")

    def test_split_traps(self):
        self.assertEqual(split_traps("fortune telling,catastrophizing"),
                         ["fortune telling", "catastrophizing"])
        self.assertEqual(split_traps(""), ["not distorted"])

    def test_patient_block(self):
        block = patient_block("I failed a test.", "I am stupid.")
        self.assertIn("I failed a test.", block)
        self.assertIn("I am stupid.", block)

    def test_screen(self):
        self.assertTrue(screen("Sometimes I want to die."))
        self.assertFalse(screen("I failed my driving test."))

    def test_negative_set_size(self):
        self.assertEqual(len(NEGATIVES), 40)


class JudgeParseTests(unittest.TestCase):
    def test_first_line(self):
        self.assertEqual(parse_verdict("NO\nIt only reframes nothing.")[0], "no")
        self.assertEqual(parse_verdict("REFRAMES\nIt offers a balanced thought.")[0], "reframes")
        self.assertEqual(parse_verdict("Unsure.")[0], "unparsed")


class StatsTests(unittest.TestCase):
    def test_holm(self):
        self.assertTrue(np.allclose(holm([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06]))
        self.assertTrue(np.isnan(holm([0.01, float("nan")])[1]))

    def test_mcnemar(self):
        a = np.array([True, False, True])
        self.assertEqual(mcnemar_exact(a, a), (0, 0, 1.0))
        only_b, only_a, p = mcnemar_exact(np.zeros(10, bool), np.ones(10, bool))
        self.assertEqual((only_b, only_a), (10, 0))
        self.assertLess(p, 0.01)

    def test_wilson(self):
        lo, hi = wilson_ci(0, 50)
        self.assertEqual(lo, 0.0)
        self.assertGreater(hi, 0.0)

    def test_kappa(self):
        self.assertAlmostEqual(cohen_kappa([1, 0, 1, 0], [1, 0, 1, 0]), 1.0)

    def test_cramers_v(self):
        self.assertAlmostEqual(cramers_v(np.array([[10, 10], [10, 10]]))[0], 0.0)
        self.assertTrue(np.isnan(cramers_v(np.array([[5, 0], [7, 0]]))[0]))


class ResumableWriterTests(unittest.TestCase):
    def test_truncated_record_is_dropped(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "runs.csv"
            fields = ["arm", "item_id", "response"]
            with ResumableWriter(path, fields, key=("arm", "item_id")) as w:
                w.write({"arm": "a", "item_id": "R000", "response": "ok"})
            with path.open("a", encoding="utf-8", newline="") as fh:
                fh.write('a,R001,"unterminated')
            with ResumableWriter(path, fields, key=("arm", "item_id")) as w:
                self.assertTrue(w.already_done("a", "R000"))
                self.assertFalse(w.already_done("a", "R001"))


if __name__ == "__main__":
    unittest.main()
