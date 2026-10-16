from chatbot.retrieval import reciprocal_rank_fusion


def test_rrf_prefers_items_in_both_lists():
    fused = reciprocal_rank_fusion(
        [
            ["a", "b", "c"],
            ["b", "a", "d"],
        ]
    )
    ids = [x[0] for x in fused]
    assert ids[0] in ("a", "b")
    assert set(ids[:3]) >= {"a", "b"}


def test_rrf_single_list_order_preserved_top():
    fused = reciprocal_rank_fusion([["x", "y", "z"]])
    assert fused[0][0] == "x"
