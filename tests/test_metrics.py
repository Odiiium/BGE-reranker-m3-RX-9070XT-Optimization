import math

import pytest

from bgeopt.metrics import MRR, NDCG, Recall, RetrievalEvaluator, build_metrics, rank_documents

# graded judgments; d4 is relevant but was never retrieved, d5 is retrieved but unjudged
QRELS = {"q1": {"d1": 3, "d2": 0, "d3": 1, "d4": 2}}
RUN = {"q1": {"d2": 0.9, "d3": 0.8, "d1": 0.7, "d5": 0.6}}
RANKED = ["d2", "d3", "d1", "d5"]


def test_rank_documents_orders_by_score_then_doc_id_desc():
    assert rank_documents(RUN["q1"]) == RANKED
    assert rank_documents({"a": 1.0, "c": 1.0, "b": 2.0}) == ["b", "c", "a"]


def test_ndcg_linear_gain_with_unretrieved_relevant_in_ideal():
    judged = QRELS["q1"]
    assert NDCG(1)(RANKED, judged) == 0.0
    dcg5 = 1 / math.log2(3) + 3 / math.log2(4)
    idcg5 = 3 / math.log2(2) + 2 / math.log2(3) + 1 / math.log2(4)
    assert NDCG(5)(RANKED, judged) == pytest.approx(dcg5 / idcg5)
    assert NDCG(10)(RANKED, judged) == pytest.approx(dcg5 / idcg5)


def test_ndcg_perfect_ranking_is_one_and_negative_grades_are_zero_gain():
    judged = {"a": 2, "b": 1, "c": -1}
    assert NDCG(3)(["a", "b", "c"], judged) == pytest.approx(1.0)
    assert NDCG(3)(["c"], judged) == 0.0


def test_mrr_respects_cutoff_and_threshold():
    judged = QRELS["q1"]
    assert MRR(1)(RANKED, judged) == 0.0
    assert MRR(5)(RANKED, judged) == pytest.approx(1 / 2)
    assert MRR(5, relevance_threshold=2)(RANKED, judged) == pytest.approx(1 / 3)
    assert MRR(2, relevance_threshold=2)(RANKED, judged) == 0.0


def test_recall_counts_all_judged_relevant():
    judged = QRELS["q1"]
    assert Recall(1)(RANKED, judged) == 0.0
    assert Recall(5)(RANKED, judged) == pytest.approx(2 / 3)
    assert Recall(5, relevance_threshold=2)(RANKED, judged) == pytest.approx(1 / 2)
    assert Recall(5)(RANKED, {"x": 0}) == 0.0


def test_build_metrics_labels_and_validation():
    labels = [m.label for m in build_metrics(["ndcg", "mrr", "recall"], [10, 1, 5])]
    assert labels == ["ndcg@1", "ndcg@5", "ndcg@10", "mrr@1", "mrr@5", "mrr@10", "recall@1", "recall@5", "recall@10"]
    with pytest.raises(ValueError):
        build_metrics(["map"], [10])


def test_evaluator_averages_over_qrels_queries_and_missing_run_scores_zero():
    qrels = {**QRELS, "q2": {"x": 1}}
    result = RetrievalEvaluator(["mrr"], [5]).evaluate(RUN, qrels)
    assert list(result.per_query.index) == ["q1", "q2"]
    assert result.per_query.loc["q2", "mrr@5"] == 0.0
    assert result.mean["mrr@5"] == pytest.approx((0.5 + 0.0) / 2)
