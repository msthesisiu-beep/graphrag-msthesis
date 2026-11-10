import importlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from common import read_jsonl, work

module = importlib.import_module("11_generate_questions")
pilot = importlib.import_module("13_pilot20")
review = importlib.import_module("14_review_questions")
optimization = importlib.import_module("15_optimization_report")
runner = importlib.import_module("07_run_configs")


class FakeRaw:
    def __init__(self, cfg):
        pass

    def check(self, passage, quote):
        return {"source_validated": quote == "The UE shall send a Registration Request.",
                "spec": passage["spec"], "clause": passage["clause"], "folder": "Rel-17/23_series",
                "md_name": "23501-ha0.md", "docx_name": "23501-ha0.docx", "quote": quote}


class QuestionGenerationTests(unittest.TestCase):
    def test_incomplete_repeat_still_runs_on_resume(self):
        cfg = self.cfg | {"kg": {"default_variant": {"triples": 10, "hops": 1, "share": 0.2}},
                          "run": {"repeats": 2}, "_config_sha256": "test-config"}
        plan = {"0": True, "A": True, "D": True, "controls": True, "modes": ["matched", "overhead"]}
        keys = runner.condition_keys(plan, ["t10-h1-g20"], "t10-h1-g20")
        module.write_jsonl(work(cfg, "runs", "dev", "log.jsonl"),
                           [{"question_id": "first", "cond": key, "run": 0, "status": "ok"} for key in keys])
        with patch.object(runner, "load_config", return_value=cfg), patch.object(runner, "Retriever"), \
                patch.object(runner, "load_split", return_value=["first"]), patch.object(runner, "get_threshold", return_value=None), \
                patch.object(runner, "load_graph"), patch.object(runner, "Builder"), \
                patch.object(runner, "warmup", side_effect=RuntimeError("pending work reached")), \
                patch("sys.argv", ["07_run_configs.py", "--plan", "pilot"]):
            with self.assertRaisesRegex(RuntimeError, "pending work reached"):
                runner.main()

    def test_kg_verification_records_time_and_tokens(self):
        kg = importlib.import_module("06_build_kg")
        triples = [{"triple_id": "triple-1", "passage_id": "source-1", "type_ok": True,
                    "subject": "UE", "relation": "sends", "object": "Request"}]
        response = {"json": {"results": [{"index": 0, "entailed": True}]}, "in_tokens": 30, "out_tokens": 10}
        with patch.object(kg, "Retriever") as retriever, patch.object(kg, "normalise", return_value=triples), \
                patch.object(kg, "warmup"), patch.object(kg, "llm", return_value=response):
            retriever.return_value.by_id = {"source-1": self.passage}
            kg.verify(self.cfg)
        rows = read_jsonl(work(self.cfg, "kg", "verify_raw.jsonl"))
        self.assertEqual(rows[0]["results"], {"triple-1": True})
        self.assertEqual(rows[0]["in_tokens"], 30)
        self.assertEqual(rows[0]["out_tokens"], 10)
        self.assertGreaterEqual(rows[0]["seconds"], 0)

    def test_completed_pilot_resume_skips_graph_retrieval_and_llm(self):
        cfg = self.cfg | {"kg": {"default_variant": {"triples": 10, "hops": 1, "share": 0.2}},
                          "run": {"repeats": 1}, "_config_sha256": "test-config"}
        plan = {"0": True, "A": True, "D": True, "controls": True, "modes": ["matched", "overhead"]}
        keys = runner.condition_keys(plan, ["t10-h1-g20"], "t10-h1-g20")
        self.assertEqual(len(keys), 11)
        module.write_jsonl(work(cfg, "runs", "dev", "log.jsonl"),
                           [{"question_id": "first", "cond": key, "run": 0, "status": "ok"} for key in keys])
        with patch.object(runner, "load_config", return_value=cfg), patch.object(runner, "Retriever"), \
                patch.object(runner, "load_split", return_value=["first"]), patch.object(runner, "get_threshold", return_value=None), \
                patch.object(runner, "load_graph") as graph, patch.object(runner, "warmup") as warmup, \
                patch.object(runner, "b_selection") as retrieval, patch("sys.argv", ["07_run_configs.py", "--plan", "pilot"]):
            runner.main()
        graph.assert_not_called()
        warmup.assert_not_called()
        retrieval.assert_not_called()

    def test_condition_keys_match_dev_grid_without_controls(self):
        plan = {"0": False, "A": False, "D": False, "controls": False, "modes": ["matched"]}
        keys = runner.condition_keys(plan, ["t5-h1-g10", "t10-h1-g20"], "t10-h1-g20")
        self.assertEqual(keys, {"B:-:-", "C:t5-h1-g10:matched", "C:t10-h1-g20:matched"})

    def test_token_count_cache_is_exact_and_tokenizer_specific(self):
        import common
        common._token_count.cache_clear()
        cfg = {"models": {"generator_hf_tokenizer": "first-tokenizer"}}
        with patch.object(common, "_tokenizer") as tokenizer_factory:
            tokenizer_factory.return_value.encode.return_value = [1, 2, 3]
            self.assertEqual(common.ntok(cfg, "same text"), 3)
            self.assertEqual(common.ntok(cfg, "same text"), 3)
            self.assertEqual(tokenizer_factory.return_value.encode.call_count, 1)
            other = {"models": {"generator_hf_tokenizer": "other-tokenizer"}}
            self.assertEqual(common.ntok(other, "same text"), 3)
            self.assertEqual(tokenizer_factory.return_value.encode.call_count, 2)
            self.assertEqual(common.ntok(cfg, ""), 0)
        common._token_count.cache_clear()

    def test_token_count_does_not_cache_large_source_documents(self):
        import common
        common._token_count.cache_clear()
        cfg = {"models": {"generator_hf_tokenizer": "tokenizer"}}
        with patch.object(common, "_tokenizer") as tokenizer_factory:
            tokenizer_factory.return_value.encode.return_value = [1]
            common.ntok(cfg, "x" * 4097)
            common.ntok(cfg, "x" * 4097)
            self.assertEqual(tokenizer_factory.return_value.encode.call_count, 2)
            self.assertEqual(common._token_count.cache_info().currsize, 0)

    def test_optimization_blocks_incomplete_or_unfair_comparisons(self):
        self.assertIn("Finish all", optimization.recommendations([], {}, 0)[0])
        conditions = [{"condition": "B:-:-", "accuracy": 0.6, "errors": 0, "missing": 0}]
        self.assertIn("budget", optimization.recommendations(conditions, {}, 1)[0].lower())
        self.assertIn("provenance", optimization.recommendations(conditions, {"B:-:-": {"provenance_ok": 0.5}}, 0)[0])

    def test_optimization_restricts_tuning_to_development_data(self):
        conditions = [{"condition": "B:-:-", "accuracy": 0.6, "errors": 0, "missing": 0},
                      {"condition": "C:t10-h1-g20:matched", "accuracy": 0.7, "errors": 0, "missing": 0}]
        advice = optimization.recommendations(conditions, {"B:-:-": {"recall_at_k": 0.5}}, 0)
        self.assertTrue(any("development" in item for item in advice))
        self.assertTrue(any("held-out test" in item for item in advice))
        self.assertEqual(optimization.finite_mean([{"value": "nan"}, {"value": "3"}], "value"), 3)

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.cfg = {"seed": 42, "paths": {"work_dir": self.temporary.name},
                    "corpus": {"release": 17, "specs": ["23.501", "24.501"], "max_tokens": 500},
                    "models": {"question_creator": "creator", "verifier": "verifier", "generator": "generator"}}
        self.passage = {"id": "source-1", "spec": "23.501", "release": 17, "clause": "5.1", "n_tokens": 200,
                        "header": "TS 23.501 clause 5.1", "text": "The UE shall send a Registration Request."}
        self.draft = {"suitable": True, "question": "Which message shall the UE send?",
                      "options": ["Registration Request", "Reject", "Release", "Accept"], "answer": 1,
                      "quote": self.passage["text"], "explanation": "The source specifies the request."}

    def response(self, result):
        return {"json": result, "truncated_risk": False, "content": "recorded"}

    def test_selection_is_order_independent_and_scoped(self):
        passages = [self.passage, self.passage | {"id": "source-2", "spec": "24.501"},
                    self.passage | {"id": "source-3", "release": 16},
                    self.passage | {"id": "source-4", "spec": "38.811"},
                    self.passage | {"id": "source-5", "n_tokens": 501}]
        selected = module.select_passages(self.cfg, passages, 5)
        self.assertEqual(selected, module.select_passages(self.cfg, list(reversed(passages)), 5))
        self.assertEqual({passage["id"] for passage in selected}, {"source-1", "source-2"})

    def test_duplicate_options_and_invented_quotes_rejected(self):
        raw = FakeRaw(self.cfg)
        with self.assertRaisesRegex(ValueError, "duplicate_options"):
            module.validate_candidate(self.cfg, self.passage, self.draft | {"options": ["Same"] * 4}, raw)
        with self.assertRaisesRegex(ValueError, "quote_not_in_both"):
            module.validate_candidate(self.cfg, self.passage, self.draft | {"quote": "Invented evidence sentence."}, raw)

    def test_generation_checkpoint_is_reused(self):
        module.write_jsonl(work(self.cfg, "passages.jsonl"), [self.passage])
        plan = {"selected_passage_ids": ["source-1"]}
        with patch.object(module, "RawEvidence", FakeRaw), patch.object(module, "llm", return_value=self.response(self.draft)) as call:
            module.generate(self.cfg, plan)
            module.generate(self.cfg, plan)
        self.assertEqual(call.call_count, 1)
        rows = read_jsonl(module.output(self.cfg, "generation.jsonl"))
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["candidate"]["dataset"], "synthetic")

    def test_blind_verification_rejects_wrong_answer_and_caches(self):
        candidate = module.validate_candidate(self.cfg, self.passage, self.draft, FakeRaw(self.cfg))
        module.write_jsonl(work(self.cfg, "passages.jsonl"), [self.passage])
        module.write_jsonl(module.output(self.cfg, "generation.jsonl"), [{"passage_id": "source-1", "status": "ok", "candidate": candidate}])
        plan = {"selected_passage_ids": ["source-1"], "target": 1}
        result = {"supported": True, "ambiguous": False, "answer": 2, "quote": self.passage["text"], "explanation": "Mismatch"}
        with patch.object(module, "RawEvidence", FakeRaw), patch.object(module, "llm", return_value=self.response(result)) as call:
            module.verify(self.cfg, plan)
            module.verify(self.cfg, plan)
        self.assertEqual(call.call_count, 1)
        messages = call.call_args.args[2]
        self.assertNotIn(self.draft["explanation"], messages[1]["content"])
        self.assertNotIn("Keyed answer", messages[1]["content"])
        self.assertEqual(module.export(self.cfg, plan), [])

    def test_export_deduplicates_and_preserves_manual_review(self):
        candidate = module.validate_candidate(self.cfg, self.passage, self.draft, FakeRaw(self.cfg))
        records = [{"question_id": "first", "status": "ok", "accepted": True, "candidate": candidate},
                   {"question_id": "second", "status": "ok", "accepted": True,
                    "candidate": candidate | {"question_id": "second"}}]
        module.write_jsonl(module.output(self.cfg, "verification.jsonl"), records)
        self.assertEqual(len(module.export(self.cfg, {"target": 3})), 1)
        path = module.output(self.cfg, "questions_review.csv")
        with path.open(newline="") as handle:
            rows = list(module.csv.DictReader(handle))
        rows[0]["manual_supported"] = "yes"
        with path.open("w", newline="") as handle:
            writer = module.csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        module.export(self.cfg, {"target": 3})
        with path.open(newline="") as handle:
            self.assertEqual(next(module.csv.DictReader(handle))["manual_supported"], "yes")

    def test_pilot_selection_is_stable_and_balanced(self):
        questions = [{"question_id": f"question-{number}", "citations": [{"spec": spec}]}
                     for number, spec in enumerate(["23.501", "23.502", "24.501", "38.300", "38.331"] * 6)]
        selected = pilot.sample_questions(questions, 20, 42)
        self.assertEqual(selected, pilot.sample_questions(list(reversed(questions)), 20, 42))
        self.assertEqual(len(selected), 20)
        for spec in ("23.501", "23.502", "24.501", "38.300", "38.331"):
            self.assertEqual(sum(question["citations"][0]["spec"] == spec for question in selected), 4)

    def test_pilot_requires_twenty_questions(self):
        with self.assertRaises(RuntimeError):
            pilot.sample_questions([], 20, 42)

    def test_pilot_accuracy_counts_failed_and_missing_answers(self):
        rows = [{"question_id": "first", "cond": "B:-:-", "run": 0, "status": "ok",
                 "valid": True, "correct": 1, "t_total": 2.0},
                {"question_id": "second", "cond": "B:-:-", "run": 0, "status": "error"}]
        summary = pilot.accuracy_summary(rows, {"first", "second", "missing"}, ["B:-:-"])[0]
        self.assertEqual(summary["correct"], 1)
        self.assertEqual(summary["errors"], 1)
        self.assertEqual(summary["missing"], 1)
        self.assertEqual(summary["accuracy"], 1 / 3)

    def test_pilot_blocks_incomplete_synthetic_generation(self):
        module.write_json(module.output(self.cfg, "plan.json"), {"selected_passage_ids": ["not-completed"]})
        with self.assertRaisesRegex(RuntimeError, "creation is incomplete"):
            pilot.completed_synthetic(self.cfg)

    def test_review_confidence_is_unavailable_before_verification(self):
        candidate = module.validate_candidate(self.cfg, self.passage, self.draft, FakeRaw(self.cfg))
        annotated = review.annotate(candidate, None)
        self.assertIsNone(annotated["confidence_score"])
        self.assertEqual(annotated["answer_rationale"], self.draft["explanation"])
        self.assertEqual(annotated["rationale_source"], "question_creator_unverified")

    def test_release_encoder_models_preserves_retrieval_inputs(self):
        from retrieval import Retriever
        retriever = Retriever.__new__(Retriever)
        retriever._models = {"embedder": object()}
        index = object()
        retriever._indexes = {"bge": index}
        retriever.by_id = {"source-1": self.passage}
        with patch("retrieval.free_memory") as release:
            retriever.release_models()
        self.assertEqual(retriever._models, {})
        self.assertIs(retriever._indexes["bge"], index)
        self.assertEqual(retriever.by_id["source-1"], self.passage)
        release.assert_called_once()

    def test_review_confidence_exposes_agreement_and_independence(self):
        candidate = module.validate_candidate(self.cfg, self.passage, self.draft, FakeRaw(self.cfg))
        citation = {"raw_md_ok": True, "raw_docx_ok": True}
        checked = candidate | {"citations": [citation, citation], "verifier_model": "verifier",
                               "verified_explanation": "The source explicitly requires this message."}
        record = {"status": "ok", "accepted": True, "candidate": checked,
                  "response": {"json": {"answer": 1, "supported": True, "ambiguous": False}}}
        self.assertEqual(review.annotate(candidate, record)["confidence_score"], 100)
        record["response"]["json"]["answer"] = 2
        self.assertEqual(review.annotate(candidate, record)["confidence_score"], 80)
        record["candidate"]["verifier_model"] = candidate["question_creator"]
        self.assertEqual(review.annotate(candidate, record)["confidence_score"], 65)


if __name__ == "__main__":
    unittest.main()