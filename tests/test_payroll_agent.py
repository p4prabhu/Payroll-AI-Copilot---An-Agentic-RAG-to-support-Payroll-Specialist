"""Offline workflow tests and explicitly enabled paid acceptance scenarios.

Normal runs skip live scenarios. After approving API charges, set
RUN_LIVE_AGENT_TESTS=1 to enable them. Three scenarios execute four turns total,
each subject to the agent's model-call and search limits. Keyword assertions
are acceptance screens, not independent verification of payroll facts.
"""

import json
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import Mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from payroll_agent import MAX_MODEL_CALLS, MAX_SEARCHES, SUBMIT_NAME, TOOL_NAME, build_agent, create_model, render_answer, submission_tool, run_turn


def final(status, text, citations=None):
    return AIMessage(content="", tool_calls=[{
        "name": SUBMIT_NAME, "id": "submit_final", "type": "tool_call",
        "args": {"status": status, "text": text, "citations": citations or []},
    }])


def call(number=1, **args):
    return AIMessage(content="", tool_calls=[{
        "name": TOOL_NAME, "args": {"question": "T4 box 22", **args},
        "id": f"call_{number}", "type": "tool_call",
    }])


def evidence():
    return {"status": "evidence_found", "results": [{
        "chunk_id": "test-chunk", "content": "Box 22 reports income tax deducted.",
        "source_url": "https://www.canada.ca/test-guide.pdf", "title": "Test government guide",
        "pdf_page_number": 18, "review_status": "pending", "needs_update_review": True,
    }]}


class FakeModel:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.seen = []

    def bind_tools(self, tools, **kwargs):
        self.tools = tools
        self.bound_options = kwargs
        return self

    def invoke(self, messages):
        self.seen.append(messages)
        return next(self.responses)


class PayrollAgentTests(unittest.TestCase):
    def test_submission_tool_is_bound_without_rejected_response_format(self):
        model = FakeModel([])
        build_agent(model, Mock())
        self.assertNotIn("response_format", model.bound_options)
        self.assertEqual(model.tools[0].name, TOOL_NAME)
        config = model.tools[1]["function"]
        self.assertEqual(config["name"], SUBMIT_NAME)
        schema = config["parameters"]
        self.assertEqual(set(schema["required"]), {"status", "text", "citations"})
        self.assertFalse(schema["additionalProperties"])

    def test_installed_integration_converts_submission_tool_for_responses_api(self):
        from langchain_perplexity import ChatPerplexity
        model = ChatPerplexity(api_key="offline-placeholder", use_responses_api=True)
        payload = model._to_responses_payload([], {"tools": [submission_tool()], "max_tokens": 1600})
        self.assertNotIn("response_format", payload)
        self.assertEqual(payload["tools"][0]["name"], SUBMIT_NAME)
        self.assertEqual(set(payload["tools"][0]["parameters"]["required"]), {"status", "text", "citations"})
        self.assertEqual(payload["max_output_tokens"], 1600)

    def test_schema_rejects_missing_wrong_type_and_extra_fields(self):
        for draft in ({"status": "clarification", "text": "Missing field"},
                      {"status": "clarification", "text": 23, "citations": []},
                      {"status": "unsupported", "text": "Gap", "citations": [], "extra": "private-text"}):
            with self.subTest(draft=draft):
                response = final("unsupported", "Placeholder")
                response.tool_calls[0]["args"] = draft
                model = FakeModel([response])
                state = run_turn(build_agent(model, Mock()), "Question")
                self.assertEqual(state["outcome"]["status"], "unsupported")
                self.assertEqual(state["trace"][-1]["reason"], "Final response schema validation failed")
                self.assertNotIn("private-text", str(state["trace"]))

    def test_incomplete_response_does_not_execute_partial_tool_call(self):
        response = call()
        response.response_metadata["status"] = "incomplete"
        retriever = Mock()
        state = run_turn(build_agent(FakeModel([response]), retriever), "T4 box 22?")
        retriever.assert_not_called()
        self.assertEqual(state["outcome"]["status"], "unsupported")
        self.assertEqual(state["trace"][-1]["reason"], "Model response was incomplete or unsuccessful")

    def test_incomplete_response_rejected_even_with_valid_submission(self):
        response = final("clarification", "What payment date?")
        response.response_metadata["status"] = "incomplete"
        state = run_turn(build_agent(FakeModel([response]), Mock()), "Question")
        self.assertEqual(state["trace"][-1]["status"], "rejected")

    def test_search_then_cited_answer_and_review_flags(self):
        retriever = Mock(return_value=evidence())
        model = FakeModel([call(edition_year=2025, document_type="guide"),
                           final("answer", "Box 22 reports income tax deducted [S1].", ["S1"])])
        state = run_turn(build_agent(model, retriever), "What is box 22?")
        self.assertEqual(state["model_calls"], 2)
        self.assertEqual(state["searches"], 1)
        self.assertIsInstance(model.seen[1][-1], ToolMessage)
        self.assertIsInstance(model.seen[0][0], SystemMessage)
        self.assertIn("Ontario Payroll", model.seen[0][0].content)
        self.assertEqual(retriever.call_args.kwargs["edition_year"], 2025)
        self.assertEqual(state["outcome"]["status"], "answer")
        answer = render_answer(state["outcome"])
        self.assertIn("PDF page 18", answer)
        self.assertIn("https://www.canada.ca/test-guide.pdf", answer)
        self.assertIn("provisional", answer)
        self.assertIn("update review", answer)

    def test_refines_empty_search(self):
        retriever = Mock(side_effect=[{"status": "no_matches", "results": []}, evidence()])
        model = FakeModel([call(1), call(2, question="Income tax deducted on T4 box 22"),
                           final("answer", "Income tax deducted.", ["S1"])])
        state = run_turn(build_agent(model, retriever), "T4 box 22?")
        self.assertEqual(state["searches"], 2)
        self.assertEqual(state["outcome"]["status"], "answer")

    def test_two_searches_preserve_distinct_evidence_and_citation_mapping(self):
        box14 = {**evidence()["results"][0], "chunk_id": "box14-chunk",
                 "content": "Box 14 reports employment income.", "pdf_page_number": 14}
        box22 = {**evidence()["results"][0], "chunk_id": "box22-chunk"}
        retriever = Mock(side_effect=[
            {"status": "evidence_found", "results": [box14]},
            {"status": "evidence_found", "results": [box22, box14]},
        ])
        model = FakeModel([
            call(1, question="T4 box 14"), call(2, question="T4 box 22"),
            final("answer", "Employment income [S1]; income tax deducted [S2].", ["S1", "S2"]),
        ])
        state = run_turn(build_agent(model, retriever), "What do T4 boxes 14 and 22 mean?")
        self.assertEqual(state["searches"], 2)
        self.assertEqual(state["outcome"]["status"], "answer")
        self.assertEqual(set(state["evidence"]), {"S1", "S2"})
        for source_id, expected in (("S1", box14), ("S2", box22)):
            self.assertEqual(state["evidence"][source_id], expected)
        second_output = json.loads(model.seen[2][-1].content)
        self.assertEqual([item["source_id"] for item in second_output["results"]], ["S2", "S1"])
        sources = {source["source_id"]: source for source in state["outcome"]["sources"]}
        self.assertEqual(sources["S1"]["pdf_page_number"], 14)
        self.assertEqual(sources["S2"]["pdf_page_number"], 18)
        rendered = render_answer(state["outcome"])
        self.assertIn("[S1] Test government guide — PDF page 14", rendered)
        self.assertIn("[S2] Test government guide — PDF page 18", rendered)

    def test_search_and_model_limits_stop_repeating_agent(self):
        retriever = Mock(return_value=evidence())
        model = FakeModel([call(i) for i in range(MAX_MODEL_CALLS)])
        state = run_turn(build_agent(model, retriever), "Keep searching")
        self.assertEqual(retriever.call_count, MAX_SEARCHES)
        self.assertEqual(state["model_calls"], MAX_MODEL_CALLS)
        self.assertEqual(state["outcome"]["status"], "unsupported")
        self.assertEqual(state["trace"][-2]["status"], "limit_reached")

    def test_parallel_requested_searches_are_executed_within_budget(self):
        calls = [call(i).tool_calls[0] for i in range(7)]
        model = FakeModel([AIMessage(content="", tool_calls=calls), final("answer", "Tax deducted.", ["S1"])])
        retriever = Mock(return_value=evidence())
        state = run_turn(build_agent(model, retriever), "Several searches")
        self.assertEqual(retriever.call_count, MAX_SEARCHES)
        self.assertEqual(len(state["evidence"]), 1)

    def test_invalid_tool_arguments_never_reach_retrieval(self):
        retriever = Mock()
        model = FakeModel([call(limit=21), final("unsupported", "Search arguments were invalid.")])
        state = run_turn(build_agent(model, retriever), "T4 box 22?")
        retriever.assert_not_called()
        self.assertEqual(state["trace"][1]["status"], "tool_error")

    def test_tool_error_is_sanitized(self):
        retriever = Mock(side_effect=RuntimeError("secret-request-detail"))
        model = FakeModel([call(), final("unsupported", "Search unavailable.")])
        state = run_turn(build_agent(model, retriever), "T4 box 22?")
        self.assertNotIn("secret-request-detail", str(state))
        self.assertEqual(state["outcome"]["status"], "unsupported")

    def test_unknown_citation_is_rejected(self):
        model = FakeModel([call(), final("answer", "Income tax deducted.", ["S999"])])
        state = run_turn(build_agent(model, Mock(return_value=evidence())), "T4 box 22?")
        self.assertEqual(state["outcome"]["status"], "unsupported")
        self.assertEqual(state["outcome"]["sources"], [])

    def test_followup_context_and_per_turn_budget_reset(self):
        model = FakeModel([final("clarification", "How often is payroll paid?"),
                           final("clarification", "What is the payment date?")])
        retriever = Mock()
        graph = build_agent(model, retriever)
        first = run_turn(graph, "What are deductions on $50,000?")
        history = [HumanMessage(content="What are deductions on $50,000?"),
                   AIMessage(content=render_answer(first["outcome"]))]
        second = run_turn(graph, "Biweekly", history)
        self.assertIn("$50,000", model.seen[1][1].content)
        self.assertEqual(second["model_calls"], 1)
        self.assertEqual(second["searches"], 0)
        retriever.assert_not_called()

    def test_previous_turn_sources_cannot_support_new_answer_without_search(self):
        model = FakeModel([call(), final("answer", "Tax deducted.", ["S1"]),
                           final("answer", "Still tax deducted.", ["S1"])])
        graph = build_agent(model, Mock(return_value=evidence()))
        first = run_turn(graph, "Box 22?")
        history = [HumanMessage(content="Box 22?"), AIMessage(content=render_answer(first["outcome"]))]
        second = run_turn(graph, "And this year?", history)
        self.assertEqual(second["evidence"], {})
        self.assertEqual(second["outcome"]["status"], "unsupported")

    def test_malformed_final_and_uncited_answer_are_rejected(self):
        for response in (AIMessage(content="not json"), final("answer", "An unsupported claim.")):
            with self.subTest(response=response.content):
                state = run_turn(build_agent(FakeModel([response]), Mock()), "Policy question")
                self.assertEqual(state["outcome"]["status"], "unsupported")

    def test_plain_json_is_not_accepted_in_place_of_submission_tool(self):
        reply = AIMessage(content=json.dumps({"status": "clarification", "text": "What date?", "citations": []}))
        state = run_turn(build_agent(FakeModel([reply]), Mock()), "Question")
        self.assertEqual(state["outcome"]["status"], "unsupported")
        self.assertEqual(state["trace"][-1]["reason"], "A single submit_response tool call is required")

    def test_submission_cannot_be_combined_with_search(self):
        combined = AIMessage(content="", tool_calls=[call().tool_calls[0], final("answer", "Claim.", ["S1"]).tool_calls[0]])
        retriever = Mock()
        state = run_turn(build_agent(FakeModel([combined]), retriever), "Question")
        retriever.assert_not_called()
        self.assertEqual(state["outcome"]["status"], "unsupported")

    def test_submission_ends_turn_without_extra_model_or_search_call(self):
        model = FakeModel([final("clarification", "What payment date?")])
        retriever = Mock()
        state = run_turn(build_agent(model, retriever), "Calculate deductions")
        self.assertEqual(len(model.seen), 1)
        retriever.assert_not_called()
        self.assertEqual(state["outcome"]["status"], "clarification")
        self.assertEqual(state["trace"][-1]["status"], "accepted")

    def test_urls_must_match_retrieved_cited_sources(self):
        for url, expected in [("https://www.canada.ca/test-guide.pdf", "answer"),
                              ("https://example.com/invented", "unsupported")]:
            with self.subTest(url=url):
                model = FakeModel([call(), final("answer", f"Tax deducted. See {url}", ["S1"])])
                state = run_turn(build_agent(model, Mock(return_value=evidence())), "Box 22?")
                self.assertEqual(state["outcome"]["status"], expected)


@unittest.skipUnless(os.getenv("RUN_LIVE_AGENT_TESTS") == "1",
                     "Paid API scenario disabled; requires explicit RUN_LIVE_AGENT_TESTS=1")
class LivePayrollAgentTests(unittest.TestCase):
    """Run only with approved API usage; each scenario builds a fresh agent."""

    def setUp(self):
        self.graph = build_agent(create_model())

    def checked_turn(self, question, history=None):
        state = run_turn(self.graph, question, history)
        outcome = state["outcome"]
        print(json.dumps({
            "input": question, "status": outcome["status"],
            "output": render_answer(outcome),
            "model_calls": state["model_calls"], "searches": state["searches"],
        }, ensure_ascii=True, indent=2), flush=True)
        self.assertEqual(state["trace"][-1]["status"], "accepted",
                         "A fallback is not a successful model response")
        self.assertLessEqual(state["model_calls"], MAX_MODEL_CALLS)
        self.assertLessEqual(state["searches"], MAX_SEARCHES)
        return state

    def test_salary_clarification_and_followup_context(self):
        question = "What are deductions on a $50,000 annual salary in Ontario?"
        first = self.checked_turn(question)
        self.assertEqual(first["outcome"]["status"], "clarification")
        self.assertIn("?", first["outcome"]["text"])
        history = [HumanMessage(content=question),
                   AIMessage(content=render_answer(first["outcome"]))]
        second = self.checked_turn(
            "Biweekly. Please confirm the annual salary I gave you and tell me what else you need.", history)
        self.assertIn(second["outcome"]["status"], ("clarification", "unsupported"))
        self.assertRegex(second["outcome"]["text"].lower(), r"50[ ,]?000|50\s*k")

    def test_t4_answer_has_evidence_for_both_boxes(self):
        state = self.checked_turn("What do T4 boxes 14 and 22 mean? Cite evidence for each and state edition limitations.")
        outcome = state["outcome"]
        self.assertEqual(outcome["status"], "answer")
        self.assertGreaterEqual(state["searches"], 1)
        # Coverage can come from one chunk or multiple; do not force extra calls.
        text = outcome["text"].lower()
        self.assertRegex(text, r"box(?:es)?\s*14")
        self.assertRegex(text, r"box(?:es)?\s*22|\b22\b")
        self.assertIn("employment income", text)
        self.assertIn("income tax", text)
        cited_content = "\n".join(source["content"] for source in outcome["sources"])
        for number in (14, 22):
            self.assertRegex(cited_content.lower(), rf"box\s+{number}\b")

    def test_employee_payday_is_reported_as_unavailable(self):
        state = self.checked_turn("When is employee Alex's next payday?")
        outcome = state["outcome"]
        self.assertEqual(outcome["status"], "unsupported")
        text = outcome["text"].lower()
        self.assertRegex(text, r"(?:employee|company|payroll|schedule|calendar|record|database)")
        self.assertRegex(text, r"(?:not|cannot|can't|unavailable|don't|do not|no access|unable)")
        self.assertNotRegex(text, r"\b\d{4}-\d{2}-\d{2}\b|\b\d{1,2}/\d{1,2}/\d{2,4}\b")


if __name__ == "__main__":
    unittest.main()
