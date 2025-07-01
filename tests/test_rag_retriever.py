from pathlib import Path

from generation_engine.rag_retriever import get_examples, refresh_index_from_session


def test_rag_retriever(tmp_path: Path) -> None:
    session_dir = tmp_path / "session_foo"
    summary_dir = session_dir / "raw" / "kaito_data"
    summary_dir.mkdir(parents=True)

    csv_path = summary_dir / "kaito_tech_summary_dummy.csv"
    csv_path.write_text(
        "trend_topic,tweet_text\n"  # header
        "LayerZero,LayerZero flipped everything 🚀\n"
        "LayerZero,Why $ZRO is pumping?\n"
    )

    refresh_index_from_session(session_dir)
    ex = get_examples("LayerZero", k=2)
    assert len(ex) == 2
    assert "LayerZero" in ex[0] 