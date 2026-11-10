import importlib
import unittest
from unittest.mock import patch

from prompts import ADJUDICATE_SCHEMA, VERIFY_SCHEMA

module = importlib.import_module("03_verify_answerability")
handoff = importlib.import_module("12_verify_then_generate")


class VerifierResponseTests(unittest.TestCase):
    def response(self, result):
        return {"json": result, "in_tokens": 10, "out_tokens": 20}

    def test_invalid_json_retries_with_larger_budget(self):
        valid = {"verdict": "no_support", "release_conflict": False, "evidence": []}
        with patch.object(module, "llm", side_effect=[self.response(None), self.response(valid)]) as call:
            result = module.verifier_response({}, [], VERIFY_SCHEMA, 1500)
        self.assertEqual([entry.kwargs["max_tokens"] for entry in call.call_args_list], [1500, 3000])
        self.assertEqual(result["in_tokens"], 20)
        self.assertEqual(result["out_tokens"], 40)

    def test_missing_adjudication_flags_cannot_silently_include(self):
        with patch.object(module, "llm", return_value=self.response({})):
            with self.assertRaisesRegex(RuntimeError, "invalid JSON"):
                module.verifier_response({}, [], ADJUDICATE_SCHEMA, 64)

    def test_invalid_boolean_is_rejected(self):
        result = {"other_option_defensible": "false", "answerable_without_standards_text": False}
        with patch.object(module, "llm", return_value=self.response(result)):
            with self.assertRaises(RuntimeError):
                module.verifier_response({}, [], ADJUDICATE_SCHEMA, 64)

    def test_valid_response_does_not_retry(self):
        result = {"other_option_defensible": False, "answerable_without_standards_text": False}
        with patch.object(module, "llm", return_value=self.response(result)) as call:
            module.verifier_response({}, [], ADJUDICATE_SCHEMA, 64)
        self.assertEqual(call.call_count, 1)

    def test_failed_batch_splits_without_dropping_passages_or_cost(self):
        question = {"question": "Which?", "options": ["One", "Two"], "answer": 1}
        passages = [{"id": "first", "header": "First", "text": "First source"},
                    {"id": "second", "header": "Second", "text": "Second source"}]
        response = self.response({"verdict": "no_support", "release_conflict": False, "evidence": []})
        with patch.object(module, "verifier_response", side_effect=[module.VerifierResponseError("cut", 30, 40), response, response]):
            results = module.verify_passages({}, question, passages)
        self.assertEqual([batch[0]["id"] for batch, _ in results], ["first", "second"])
        self.assertEqual(sum(result["in_tokens"] for _, result in results), 50)
        self.assertEqual(sum(result["out_tokens"] for _, result in results), 80)

    def test_single_passage_failure_remains_an_error(self):
        question = {"question": "Which?", "options": ["One", "Two"], "answer": 1}
        with patch.object(module, "verifier_response", side_effect=module.VerifierResponseError("cut", 1, 2)):
            with self.assertRaises(module.VerifierResponseError):
                module.verify_passages({}, question, [{"id": "first", "header": "First", "text": "Source"}])

    def test_handoff_blocks_missing_and_failed_records(self):
        questions = [{"question_id": "first"}, {"question_id": "second"}]
        with self.assertRaises(RuntimeError):
            handoff.require_complete(questions, [{"question_id": "first", "status": "ok"}])
        with self.assertRaises(RuntimeError):
            handoff.require_complete(questions, [{"question_id": "first", "status": "ok"},
                                                 {"question_id": "second", "status": "error"}])
        handoff.require_complete(questions, [{"question_id": "first", "status": "ok"},
                                             {"question_id": "second", "status": "ok"}])


if __name__ == "__main__":
    unittest.main()