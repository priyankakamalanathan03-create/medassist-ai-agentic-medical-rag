import unittest
import os
import sqlite3
import tempfile
from types import SimpleNamespace
from unittest.mock import Mock, patch

from agents.base_agent import BaseAgent
from agents.critique_agent import CritiqueAgent
from agents.orchestrator import OrchestratorAgent
from agents.rag_agent import rag_agent
from agents.symptom_agent import symptom_agent
import analytics
from llm.groq_client import GroqLLMClient, GroqRateLimitError
from rag.embeddings import load_embedding_model
from rag.retriever import HybridRetriever
from tools.summarize_tool import _summarize_document
from tools.search_tool import _search_documents
from tools.symptom_checker_tool import _check_symptoms
from tools.tool_registry import Tool, ToolRegistry


class TestIntentRouting(unittest.TestCase):
    def setUp(self):
        self.orchestrator = OrchestratorAgent()

    def test_summary_routes_to_summarization_without_an_llm_call(self):
        self.assertEqual(
            self.orchestrator._classify_intent("Summarize the key findings about asthma."),
            "summarization",
        )
        self.assertEqual(
            self.orchestrator.INTENT_MAP["summarization"].agent_name,
            "RAG Medical Agent",
        )
        self.assertNotIn("general_medical", self.orchestrator.INTENT_MAP)

    def test_personal_symptoms_route_to_symptom_agent(self):
        self.assertEqual(
            self.orchestrator._classify_intent(
                "I have shortness of breath and wheezing. What could this indicate?"
            ),
            "symptom_analysis",
        )

    def test_medical_fact_question_routes_to_document_search(self):
        self.assertEqual(
            self.orchestrator._classify_intent("What are the symptoms of asthma?"),
            "document_qa",
        )


class TestAgentTools(unittest.TestCase):
    def test_embedding_model_load_uses_local_cache(self):
        model = Mock()
        with patch("rag.embeddings.SentenceTransformer", return_value=model) as loader:
            loaded_model = load_embedding_model()

        self.assertIs(loaded_model, model)
        loader.assert_called_once()
        self.assertTrue(loader.call_args.kwargs["local_files_only"])

    def test_tool_registry_rejects_missing_required_arguments(self):
        registry = ToolRegistry()
        called = []
        registry.register(
            Tool(
                name="search_documents",
                description="Search documents.",
                func=lambda query: called.append(query) or {"results": []},
                parameters={"query": "Search phrase"},
            )
        )

        result = registry.call("search_documents")

        self.assertIn("Invalid arguments", result["error"])
        self.assertIn("query", result["error"])
        self.assertEqual(called, [])

    def test_search_tool_clamps_result_count_and_returns_citations(self):
        calls = []

        def retrieve(query, k):
            calls.append((query, k))
            return [
                {"text": "Insulin regulates glucose.", "source": "Diabetes.pdf", "page": 1}
            ]

        retriever_stub = SimpleNamespace(
            retrieve=retrieve,
        )
        with patch("rag.retriever.retriever", retriever_stub):
            result = _search_documents("insulin", k=99)

        self.assertEqual(result["num_results"], 1)
        self.assertEqual(calls, [("insulin", 10)])
        self.assertEqual(result["results"][0]["source"], "Diabetes.pdf")
        self.assertEqual(result["results"][0]["page"], 1)

    def test_summary_tool_returns_page_citations_that_agents_can_collect(self):
        retriever_stub = SimpleNamespace(
            documents=[
                {"source": "Diabetes.pdf", "page": 1, "chunk_id": "a", "category": "Endocrine", "text": "Insulin regulates glucose."},
                {"source": "Diabetes.pdf", "page": 2, "chunk_id": "b", "category": "Endocrine", "text": "Type 1 diabetes needs insulin."},
            ]
        )
        with patch("rag.retriever.retriever", retriever_stub):
            result = _summarize_document("Diabetes.pdf")

        self.assertIn("[Diabetes.pdf, p. 1]", result["summary_chunks"][0])
        sources = []
        BaseAgent._collect_sources(result, sources)
        self.assertEqual(sources, ["Diabetes.pdf (p. 1)", "Diabetes.pdf (p. 2)"])

    def test_symptom_tool_reports_keyword_match_not_clinical_confidence(self):
        retriever_stub = SimpleNamespace(
            retrieve=lambda query, k: [
                {"source": "Asthma.pdf", "page": 2, "text": "Wheezing may occur in asthma."}
            ]
        )
        with patch("rag.retriever.retriever", retriever_stub):
            result = _check_symptoms("shortness of breath, wheezing")

        self.assertIn("possible_conditions", result)
        self.assertIn("keyword_match", result["possible_conditions"][0])
        self.assertNotIn("confidence", result["possible_conditions"][0])
        self.assertIn("not a diagnosis", result["interpretation_note"])
        self.assertTrue(result["possible_conditions"][0]["evidence"])

    def test_tool_errors_are_explicitly_marked_as_non_evidence(self):
        message = BaseAgent._observation_message(
            {"error": "query is required"},
            '{"error": "query is required"}',
        )

        self.assertIn("returned no evidence", message)
        self.assertIn("never treat this error as retrieved information", message)

    def test_retriever_starts_with_empty_corpus_when_no_local_data_is_packaged(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            with (
                patch("rag.retriever.DOCUMENTS_FILE", os.path.join(temp_dir, "documents.pkl")),
                patch("rag.retriever.FAISS_FILE", os.path.join(temp_dir, "index.bin")),
                patch("rag.retriever.BM25_FILE", os.path.join(temp_dir, "bm25.pkl")),
                patch(
                    "rag.retriever.load_embedding_model",
                    return_value=SimpleNamespace(encode=Mock()),
                ),
            ):
                retriever = HybridRetriever()

        self.assertEqual(retriever.documents, [])
        self.assertEqual(retriever.retrieve("asthma symptoms"), [])


class TestGroqRateLimits(unittest.TestCase):
    @staticmethod
    def make_client(create_mock):
        client = object.__new__(GroqLLMClient)
        client.model = "test-model"
        client.client = SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(create=create_mock),
            )
        )
        client._last_call_time = 0
        client._rate_limited_until = 0.0
        client._daily_quota_exhausted = False
        return client

    def test_daily_token_limit_is_reported_without_waiting_or_retrying(self):
        provider_error = RuntimeError(
            "429 tokens per day (TPD) limit reached. "
            "Please try again in 35m25.44s."
        )
        provider_error.status_code = 429
        create = Mock(side_effect=provider_error)
        client = self.make_client(create)

        with patch("llm.groq_client.time.monotonic", return_value=1000.0):
            with self.assertRaisesRegex(
                GroqRateLimitError,
                r"daily token quota.*35m 25s",
            ):
                client.generate([{"role": "user", "content": "test"}])

            with self.assertRaisesRegex(
                GroqRateLimitError,
                "daily token quota",
            ):
                client.generate([{"role": "user", "content": "test again"}])
            self.assertTrue(client.get_rate_limit_status()["limited"])

        create.assert_called_once()

    def test_streaming_rate_limit_is_not_retried(self):
        provider_error = RuntimeError(
            "429 rate_limit_exceeded; try again in 12s."
        )
        provider_error.status_code = 429
        create = Mock(side_effect=provider_error)
        client = self.make_client(create)

        with self.assertRaisesRegex(GroqRateLimitError, "rate limit reached"):
            list(client.generate_stream([{"role": "user", "content": "test"}]))

        create.assert_called_once()

    def test_streaming_certificate_failure_is_actionable_and_not_retried(self):
        provider_error = RuntimeError("Connection error.")
        provider_error.__cause__ = RuntimeError(
            "[SSL: CERTIFICATE_VERIFY_FAILED] unable to get local issuer certificate"
        )
        create = Mock(side_effect=provider_error)
        client = self.make_client(create)

        with self.assertRaisesRegex(
            RuntimeError,
            "TLS certificate validation failed while connecting to Groq",
        ):
            list(client.generate_stream([{"role": "user", "content": "test"}]))

        create.assert_called_once()

    def test_missing_api_key_never_echoes_the_user_prompt_as_an_answer(self):
        client = self.make_client(Mock())
        client.client = None

        with self.assertRaisesRegex(EnvironmentError, "GROQ_API_KEY not set"):
            client.generate([{"role": "user", "content": "give me medical advice"}])


class TestCitationValidation(unittest.TestCase):
    def setUp(self):
        self.critic = CritiqueAgent()

    def test_supported_answer_scores_high_and_explains_metrics(self):
        result = self.critic.critique(
            query="What are common symptoms of asthma?",
            draft_answer=(
                "Common asthma symptoms include cough and wheezing according to the indexed "
                "document [Asthma.pdf, p. 2]."
            ),
            sources=["Asthma.pdf (p. 2)"],
        )
        self.assertEqual(result["quality_score"], "High")
        self.assertIn("Medical Disclaimer", result["validated_answer"])
        metrics = result["quality_metrics"]["answer_quality"]
        self.assertEqual(metrics["metrics"]["citation_coverage"], 100)
        self.assertEqual(metrics["metrics"]["retrieved_source_match"], 100)
        self.assertEqual(sum(metrics["weights"].values()), 100)
        self.assertIn("not a fact-check", metrics["note"])
        self.assertIn("does not check whether the passage supports the claim", metrics["note"])

    def test_unmatched_citation_lowers_score_and_is_explained(self):
        result = self.critic.critique(
            query="What does the document report about asthma symptoms?",
            draft_answer="The document reports symptoms and breathing problems [Other.pdf, p. 1].",
            sources=["Asthma.pdf (p. 2)"],
        )
        self.assertNotEqual(result["quality_score"], "High")
        self.assertEqual(
            result["quality_metrics"]["answer_quality"]["metrics"]["retrieved_source_match"],
            0,
        )
        self.assertTrue(result["quality_metrics"]["answer_quality"]["suggestions"])

    def test_missing_citations_result_in_low_answer_evidence_score(self):
        result = self.critic.critique(
            query="What does the medical document report?",
            draft_answer="An unsupported medical statement.",
            sources=[],
        )
        self.assertEqual(result["quality_score"], "Low")
        self.assertEqual(
            result["quality_metrics"]["answer_quality"]["metrics"]["citation_coverage"],
            0,
        )

    def test_prompt_clarity_is_separate_and_does_not_block_answer(self):
        weak_prompt = self.critic.critique(
            query="Asthma?",
            draft_answer=(
                "Asthma symptoms can include wheezing, coughing, and chest tightness "
                "according to the indexed document [Asthma.pdf, p. 2]."
            ),
            sources=["Asthma.pdf (p. 2)"],
        )
        moderate_prompt = self.critic.critique(
            query="Explain asthma symptoms.",
            draft_answer="Asthma symptoms can include coughing [Asthma.pdf, p. 2].",
            sources=["Asthma.pdf (p. 2)"],
        )
        clear_prompt = self.critic.critique(
            query=(
                "Summarize common asthma symptoms from the document, include page citations "
                "and present the answer as a short bullet list."
            ),
            draft_answer="Asthma can cause wheezing [Asthma.pdf, p. 2].",
            sources=["Asthma.pdf (p. 2)"],
        )
        self.assertLess(
            weak_prompt["quality_metrics"]["prompt_quality"]["score"],
            clear_prompt["quality_metrics"]["prompt_quality"]["score"],
        )
        self.assertTrue(weak_prompt["validated_answer"])
        self.assertEqual(weak_prompt["quality_metrics"]["answer_quality"]["level"], "High")
        self.assertEqual(weak_prompt["quality_metrics"]["prompt_quality"]["level"], "Low")
        self.assertEqual(
            moderate_prompt["quality_metrics"]["prompt_quality"]["level"],
            "Moderate",
        )
        self.assertEqual(clear_prompt["quality_metrics"]["prompt_quality"]["level"], "High")
        self.assertIn(
            "never blocks an answer",
            weak_prompt["quality_metrics"]["prompt_quality"]["note"],
        )

    def test_citation_page_must_match_retrieved_page(self):
        result = self.critic.critique(
            query="What are asthma symptoms?",
            draft_answer="Asthma symptoms include coughing and wheezing [Asthma.pdf, p. 4].",
            sources=["Asthma.pdf (p. 2)"],
        )
        self.assertEqual(
            result["quality_metrics"]["answer_quality"]["metrics"]["retrieved_source_match"],
            0,
        )

    def test_source_match_accepts_parentheses_in_pdf_filename(self):
        result = self.critic.critique(
            query="What does the document report about cardiovascular disease?",
            draft_answer=(
                "The document discusses cardiovascular disease "
                "[Cardiovascular diseases (CVDs).pdf, p. 4]."
            ),
            sources=["Cardiovascular diseases (CVDs).pdf (p. 4)"],
        )

        self.assertEqual(
            result["quality_metrics"]["answer_quality"]["metrics"]["retrieved_source_match"],
            100,
        )

    def test_user_symptom_restating_and_emergency_notice_are_not_document_claims(self):
        answer = (
            "**URGENT MEDICAL ALERT: SEEK EMERGENCY CARE IMMEDIATELY. "
            "Call emergency services; do not wait for an AI assessment.**\n\n"
            "## Symptom Analysis\n"
            "Patient symptoms: chest pain, shortness of breath, and fatigue.\n\n"
            "## Possible Conditions\n"
            "Asthma is a chronic lung disease involving airway inflammation "
            "[Asthma.pdf, p. 1].\n"
            "Uncited factual claims should still be detected."
        )
        result = self.critic.critique(
            query="I have chest pain and shortness of breath. What could this be?",
            draft_answer=answer,
            sources=["Asthma.pdf (p. 1)"],
        )
        answer_quality = result["quality_metrics"]["answer_quality"]

        self.assertEqual(answer_quality["metrics"]["citation_coverage"], 50)
        self.assertIn(
            "Add inline file-and-page citations to factual statements.",
            answer_quality["suggestions"],
        )
        self.assertIn("urgent-care safety notices", answer_quality["note"])

    def test_partial_source_match_produces_moderate_answer_score(self):
        result = self.critic.critique(
            query="What are asthma symptoms?",
            draft_answer=(
                "Asthma symptoms may include coughing and wheezing, as reported "
                "in the indexed document "
                "[Other.pdf, p. 5]."
            ),
            sources=["Asthma.pdf (p. 2)"],
        )
        answer_quality = result["quality_metrics"]["answer_quality"]
        self.assertEqual(answer_quality["level"], "Moderate")
        self.assertEqual(answer_quality["metrics"]["retrieved_source_match"], 0)

    def test_instruction_words_and_citations_do_not_reduce_topic_relevance(self):
        result = self.critic.critique(
            query=(
                "Explain cancer using Cancer.pdf in two bullet points. Cite the exact page "
                "for each factual statement."
            ),
            draft_answer=(
                "## Answer\n"
                "Cancer involves abnormal cells that invade other body parts "
                "[Cancer.pdf, p. 1].\n\n"
                "## Sources\nCancer.pdf\n\n"
                "## Medical Disclaimer\nThis is educational information."
            ),
            sources=["Cancer.pdf (p. 1)"],
        )
        self.assertEqual(
            result["quality_metrics"]["answer_quality"]["metrics"]["query_term_relevance"],
            100,
        )

    def test_topic_relevance_matches_common_diagnosis_and_management_word_forms(self):
        result = self.critic.critique(
            query=(
                "How is asthma diagnosed and managed according to clinical guidelines? "
                "and summarizze in 3 bullet points"
            ),
            draft_answer=(
                "## Answer\n"
                "- The protocol describes assessment and diagnosis of asthma "
                "[Asthma.pdf, p. 4].\n"
                "- Management includes inhaled medication to control asthma "
                "[Asthma.pdf, p. 1].\n"
                "## Medical Disclaimer\nThis is educational information."
            ),
            sources=["Asthma.pdf (p. 1)", "Asthma.pdf (p. 4)"],
        )
        answer_quality = result["quality_metrics"]["answer_quality"]
        self.assertEqual(answer_quality["metrics"]["query_term_relevance"], 60)
        self.assertNotIn(
            "Address the main topic terms in the question more directly.",
            answer_quality["suggestions"],
        )

    def test_unreferenced_intro_is_flagged_but_per_sentence_citations_clear_it(self):
        answer = (
            'The indexed documents do not define "endocrine." The Diabetes document '
            "discusses the pancreas and insulin production. "
            "The endocrine system involves glands that produce hormones.\n\n"
            "The pancreas produces insulin [Diabetes.pdf, p. 1]."
        )
        result = self.critic.critique(
            query="What does the document say about endocrine?",
            draft_answer=answer,
            sources=["Diabetes.pdf (p. 1)"],
        )
        answer_quality = result["quality_metrics"]["answer_quality"]
        self.assertEqual(answer_quality["metrics"]["citation_coverage"], 25)
        self.assertIn(
            "Add inline file-and-page citations to factual statements.",
            answer_quality["suggestions"],
        )

        cited_answer = (
            'The indexed documents do not define "endocrine" [Diabetes.pdf, p. 1]. '
            "The Diabetes document discusses the pancreas and insulin production "
            "[Diabetes.pdf, p. 1]. "
            "The pancreas produces insulin [Diabetes.pdf, p. 1]."
        )
        cited_result = self.critic.critique(
            query="What does the document say about endocrine?",
            draft_answer=cited_answer,
            sources=["Diabetes.pdf (p. 1)"],
        )
        cited_quality = cited_result["quality_metrics"]["answer_quality"]
        self.assertEqual(cited_quality["metrics"]["citation_coverage"], 100)
        self.assertNotIn(
            "Add inline file-and-page citations to factual statements.",
            cited_quality["suggestions"],
        )

    def test_medical_agent_prompts_require_citations_on_each_factual_sentence(self):
        for agent in (rag_agent, symptom_agent):
            with self.subTest(agent=agent.agent_name):
                self.assertIn("sentence by sentence", agent.system_prompt)
                self.assertIn("Sources", agent.system_prompt)
                self.assertIn("uncited sentence", agent.system_prompt)
                self.assertIn(
                    "filename-and-page combinations that appear in the retrieved passages",
                    agent.system_prompt,
                )


class TestQualityAnalytics(unittest.TestCase):
    def test_existing_database_migrates_and_tracks_both_scores(self):
        with tempfile.TemporaryDirectory() as directory:
            database_path = os.path.join(directory, "analytics.db")
            connection = sqlite3.connect(database_path)
            connection.execute(
                """
                CREATE TABLE query_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    query TEXT NOT NULL,
                    model TEXT,
                    token_count INTEGER,
                    intent TEXT,
                    quality INTEGER
                )
                """
            )
            connection.execute(
                """
                INSERT INTO query_log (session_id, timestamp, query, quality)
                VALUES ('legacy', '2024-01-01', 'Old scoring record', 3)
                """
            )
            connection.commit()
            connection.close()

            with patch.object(analytics, "DB_PATH", database_path):
                analytics._ensure_schema()
                analytics.log_query(
                    session_id="test",
                    query="What are asthma symptoms?",
                    quality=3,
                    answer_score=92,
                    prompt_score=65,
                )
                stats = analytics.get_stats()

            self.assertEqual(stats["avg_answer_score"], 92)
            self.assertEqual(stats["avg_prompt_score"], 65)
            self.assertEqual(stats["quality_breakdown"]["high"], 1)


class StreamingTestAgent(BaseAgent):
    MAX_ITERATIONS = 2

    @property
    def agent_name(self):
        return "Test Agent"

    @property
    def system_prompt(self):
        return "Use the search tool and cite the source."


class TestAgentStreaming(unittest.TestCase):
    def test_final_answer_streams_with_retrieved_citations(self):
        def fake_stream(messages, **kwargs):
            if any(message["content"].startswith("Observation:") for message in messages):
                yield "Thought: I have enough evidence.\nFinal Answer:\n"
                yield "The indexed document supports this claim [Asthma.pdf, p. 2]."
            else:
                yield 'Thought: Search the index.\nAction: search_documents\nAction Input: {"query": "asthma", "k": 1}'

        with (
            patch("agents.base_agent.llm_client.generate_stream", side_effect=fake_stream),
            patch(
                "agents.base_agent.tool_registry.call",
                return_value={
                    "results": [
                        {"source": "Asthma.pdf", "page": 2, "text": "Supporting passage"}
                    ]
                },
            ),
        ):
            events = list(StreamingTestAgent().run_stream("What does the document say?"))

        streamed_text = "".join(
            event["data"] for event in events if event["type"] == "token"
        )
        completed = next(event["data"] for event in events if event["type"] == "done")
        self.assertIn("supports this claim", streamed_text)
        self.assertEqual(completed["sources"], ["Asthma.pdf (p. 2)"])

    def test_history_answer_gets_a_fresh_source_lookup(self):
        def fake_stream(messages, **kwargs):
            yield "Final Answer:\nThe symptoms may worsen at night [Asthma.pdf, p. 2]."

        with (
            patch("agents.base_agent.llm_client.generate_stream", side_effect=fake_stream),
            patch(
                "agents.base_agent.tool_registry.call",
                return_value={
                    "results": [
                        {"source": "Asthma.pdf", "page": 2, "text": "Symptoms often worsen at night"}
                    ]
                },
            ) as search,
        ):
            events = list(StreamingTestAgent().run_stream("Which symptoms worsen at night?"))

        completed = next(event["data"] for event in events if event["type"] == "done")
        self.assertEqual(completed["sources"], ["Asthma.pdf (p. 2)"])
        search.assert_called_once_with(
            "search_documents",
            query="Which symptoms worsen at night?",
            k=5,
        )


if __name__ == "__main__":
    unittest.main()
