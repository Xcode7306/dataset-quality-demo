"""v0.9 标准依据 RAG、引用绑定和规则编制集成测试。"""

from datetime import date
import json
from pathlib import Path
import unittest

from jsonschema import Draft202012Validator

from src.rag.citations import RagCitationError, evidence_from_response
from src.rag.ingestion import RagIngestionError, ingest_document_bytes
from src.rag.models import RAG_NAMESPACE_DATA_DICTIONARY, RAG_NAMESPACE_STANDARDS
from src.rag.retrieval import RagKnowledgeBase
from src.rule_authoring_service import (
    build_rule_pack_from_draft,
    compile_rule_draft,
    validate_rule_draft,
)
from src.rule_authoring_workflow import RuleAuthoringWorkflow
from src.rule_dsl import RuleDraftValidationError, make_workflow_id
from src.rule_pack import draft_sha256
from src.upload_service import evaluate_uploaded_dataset
from src.workflow import build_profile_report


ROOT = Path(__file__).resolve().parents[1]
REFERENCE_DATE = date(2026, 7, 17)
SAMPLE = ROOT / "sample_data" / "good_dataset.csv"

RAG_TEXT_V1_TEXT = """# DB31/T 1523-2024 公共数据质量评价要求
版本：v1
发布日期：2024-06-01

## 2 数据规范

### 2.1 服务名称
指标代码 020100。service_name 服务名称必须填写，空值不得作为有效数据。
"""
RAG_TEXT_V1 = RAG_TEXT_V1_TEXT.encode("utf-8")

RAG_TEXT_V2 = RAG_TEXT_V1_TEXT.replace(
    "版本：v1", "版本：v2"
).replace(
    "必须填写，空值不得作为有效数据", "必须填写，且应与目录服务清单一致"
).encode("utf-8")

LOCAL_RAG_TEXT = """# Local Test Standard
版本：v1

## 1 字段要求
local_field 必须填写。
""".encode("utf-8")


class RagV09Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report = build_profile_report(SAMPLE, reference_date=REFERENCE_DATE)
        cls.report_for_custom = evaluate_uploaded_dataset(
            (
                b"status,service_name\n"
                b"active,alpha\n"
                b"inactive,beta\n"
            ),
            "rag-custom.csv",
            reference_date=REFERENCE_DATE,
        )
        cls.draft_schema = json.loads(
            (ROOT / "schemas" / "rule-draft.schema.json").read_text(encoding="utf-8")
        )
        cls.pack_schema = json.loads(
            (ROOT / "schemas" / "rule-pack.schema.json").read_text(encoding="utf-8")
        )
        cls.document_schema = json.loads(
            (ROOT / "schemas" / "rag-document.schema.json").read_text(encoding="utf-8")
        )
        cls.response_schema = json.loads(
            (ROOT / "schemas" / "rag-search-response.schema.json").read_text(encoding="utf-8")
        )
        for schema in (
            cls.draft_schema,
            cls.pack_schema,
            cls.document_schema,
            cls.response_schema,
        ):
            Draft202012Validator.check_schema(schema)

    def test_ingestion_preserves_metadata_and_stable_chunk_locator(self):
        first = ingest_document_bytes(
            RAG_TEXT_V1,
            "standard-v1.md",
            source_namespace=RAG_NAMESPACE_STANDARDS,
            approved=True,
        )
        second = ingest_document_bytes(
            RAG_TEXT_V1,
            "standard-v1.md",
            source_namespace=RAG_NAMESPACE_STANDARDS,
            approved=True,
        )

        self.assertEqual(first.document.standard_number, "DB31/T1523-2024")
        self.assertEqual(first.document.version, "v1")
        self.assertEqual(first.document.effective_status, "active")
        self.assertEqual(first.document.chunk_ids, second.document.chunk_ids)
        self.assertTrue(any("db31_020100" in chunk.metric_ids for chunk in first.chunks))
        clause_chunk = next(
            chunk for chunk in first.chunks if "service_name" in chunk.text
        )
        self.assertEqual(clause_chunk.section, "2.1 服务名称")
        self.assertGreaterEqual(clause_chunk.line_start, 1)
        self.assertTrue(
            Draft202012Validator(self.document_schema).is_valid(
                first.document.to_dict()
            )
        )

    def test_retrieval_filters_before_ranking_and_surfaces_version_conflict(self):
        knowledge_base = RagKnowledgeBase()
        knowledge_base.ingest_bytes(
            RAG_TEXT_V1,
            "standard-v1.md",
            source_namespace=RAG_NAMESPACE_STANDARDS,
            approved=True,
        )
        knowledge_base.ingest_bytes(
            RAG_TEXT_V2,
            "standard-v2.md",
            source_namespace=RAG_NAMESPACE_STANDARDS,
            approved=True,
        )
        knowledge_base.ingest_bytes(
            RAG_TEXT_V1,
            "dictionary.md",
            source_namespace=RAG_NAMESPACE_DATA_DICTIONARY,
            approved=False,
        )

        conflict = knowledge_base.search(
            "service_name 必须填写",
            metric_id="db31_020100",
            standard_number="DB31/T 1523-2024",
        )
        self.assertEqual(conflict.status, "conflict")
        self.assertIsNotNone(conflict.conflict)
        self.assertEqual(conflict.filtered_document_count, 2)
        self.assertEqual(
            set(conflict.conflict.versions),  # type: ignore[union-attr]
            {"v1", "v2"},
        )
        self.assertTrue(
            Draft202012Validator(self.response_schema).is_valid(
                conflict.to_dict(include_text=True)
            )
        )

        selected = knowledge_base.search(
            "service_name 必须填写",
            metric_id="db31_020100",
            standard_number="DB31/T 1523-2024",
            version="v1",
        )
        self.assertEqual(selected.status, "ok")
        self.assertTrue(selected.results)
        self.assertEqual(selected.version, "v1")
        empty = knowledge_base.search("zzqwerty_nonexistent")
        self.assertEqual(empty.status, "no_results")
        self.assertEqual(empty.results, ())

    def test_unapproved_and_expired_sources_are_not_retrievable_or_bindable(self):
        knowledge_base = RagKnowledgeBase()
        knowledge_base.ingest_bytes(
            RAG_TEXT_V1,
            "active-v1.md",
            source_namespace=RAG_NAMESPACE_STANDARDS,
            approved=True,
            effective_status="active",
        )
        knowledge_base.ingest_bytes(
            RAG_TEXT_V1,
            "expired-v1.md",
            source_namespace=RAG_NAMESPACE_STANDARDS,
            approved=True,
            effective_status="expired",
        )
        knowledge_base.ingest_bytes(
            RAG_TEXT_V1,
            "unapproved-v1.md",
            source_namespace=RAG_NAMESPACE_STANDARDS,
            approved=False,
            effective_status="active",
        )

        response = knowledge_base.search(
            "service_name 必须填写",
            metric_id="db31_020100",
            standard_number="DB31/T 1523-2024",
            version="v1",
        )
        self.assertEqual(response.status, "ok")
        self.assertEqual(response.filtered_document_count, 1)
        self.assertTrue(response.results)
        self.assertTrue(
            all(
                item.document.approved
                and item.document.effective_status == "active"
                for item in response.results
            )
        )
        evidence = evidence_from_response(response)
        self.assertTrue(evidence)
        self.assertTrue(all(item.authoritative for item in evidence))

    def test_citations_are_bound_to_retrieval_results_and_pack_source(self):
        knowledge_base = RagKnowledgeBase()
        knowledge_base.ingest_bytes(
            RAG_TEXT_V1,
            "standard-v1.md",
            source_namespace=RAG_NAMESPACE_STANDARDS,
            approved=True,
        )
        response = knowledge_base.search(
            "service_name 必须填写",
            metric_id="db31_020100",
            version="v1",
        )
        result = response.results[0]
        evidence = evidence_from_response(
            response,
            selected_chunk_ids=(result.chunk.chunk_id,),
        )
        self.assertEqual(len(evidence), 1)
        self.assertEqual(evidence[0].type, "standard_clause")
        self.assertEqual(evidence[0].source_id, result.chunk.chunk_id)
        self.assertEqual(evidence[0].document_version, "v1")
        self.assertEqual(evidence[0].chunk_id, result.chunk.chunk_id)

        draft = compile_rule_draft(
            self.report,
            target_metric_id="db31_020100",
            user_intent="service_name为必填字段",
            rag_response=response,
            selected_chunk_ids=(result.chunk.chunk_id,),
            created_at="2026-08-06T00:00:00Z",
        )
        self.assertTrue(
            any(item.type == "standard_clause" for item in draft.evidence)
        )
        validation = validate_rule_draft(draft, self.report)
        self.assertTrue(validation.valid, validation.errors)
        self.assertTrue(
            Draft202012Validator(self.draft_schema).is_valid(draft.to_dict())
        )

        pack = build_rule_pack_from_draft(draft, self.report)
        self.assertEqual(pack.source.type, "standard_retrieval")
        self.assertEqual(pack.source.generator, "quality-rule-agent-v0.9")
        self.assertEqual(pack.version, "0.9.0")
        self.assertTrue(any(item["type"] == "standard_clause" for item in pack.evidence))
        self.assertEqual(draft_sha256(pack), draft_sha256(pack))
        self.assertTrue(
            Draft202012Validator(self.pack_schema).is_valid(pack.to_dict())
        )

    def test_conflicts_and_missing_ids_cannot_be_bound(self):
        knowledge_base = RagKnowledgeBase()
        knowledge_base.ingest_bytes(
            RAG_TEXT_V1,
            "standard-v1.md",
            source_namespace=RAG_NAMESPACE_STANDARDS,
            approved=True,
        )
        knowledge_base.ingest_bytes(
            RAG_TEXT_V2,
            "standard-v2.md",
            source_namespace=RAG_NAMESPACE_STANDARDS,
            approved=True,
        )
        response = knowledge_base.search("service_name 必须填写")
        self.assertEqual(response.status, "conflict")
        with self.assertRaises(RagCitationError):
            evidence_from_response(
                response,
                selected_chunk_ids=(response.results[0].chunk.chunk_id,),
            )
        with self.assertRaises(RuleDraftValidationError):
            compile_rule_draft(
                self.report,
                target_metric_id="db31_020100",
                user_intent="service_name为必填字段",
                rag_response=response,
                selected_chunk_ids=("chunk-not-from-response",),
            )

    def test_no_retrieval_source_does_not_create_standard_claim(self):
        draft = compile_rule_draft(
            self.report,
            target_metric_id="db31_020100",
            user_intent="service_name为必填字段",
            created_at="2026-08-06T00:00:00Z",
        )
        self.assertFalse(
            any(
                item.type in {"standard_clause", "data_dictionary"}
                for item in draft.evidence
            )
        )
        pack = build_rule_pack_from_draft(draft, self.report)
        self.assertEqual(pack.source.type, "user_natural_language")

    def test_plain_text_pdf_boundary_and_workflow_retrieving_state(self):
        plain = ingest_document_bytes(
            "服务名称必须填写。".encode("utf-8"),
            "plain.txt",
            source_namespace=RAG_NAMESPACE_STANDARDS,
            approved=True,
        )
        self.assertTrue(plain.chunks)
        workflow = RuleAuthoringWorkflow(
            workflow_id=make_workflow_id("rag-test"),
            target_metric_id="db31_020100",
        )
        self.assertEqual(workflow.start_retrieving().state, "retrieving")
        self.assertEqual(
            workflow.start_retrieving().start_compiling().state,
            "compiling",
        )
        with self.assertRaises(RagIngestionError):
            ingest_document_bytes(
                b"not-a-pdf",
                "unsupported.pdf",
                source_namespace=RAG_NAMESPACE_STANDARDS,
                approved=True,
            )

if __name__ == "__main__":
    unittest.main()
