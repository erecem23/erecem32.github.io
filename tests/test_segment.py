from reinert.preprocessing.segment import segment_text


def test_sentence_segmentation() -> None:
    text = "Uno. Dos? Tres! Cuatro;"
    segments = segment_text(text, mode="sentence")
    assert segments == ["Uno", "Dos", "Tres", "Cuatro"]


def test_fixed_segmentation() -> None:
    text = "a b c d e"
    segments = segment_text(text, mode="fixed", segment_size=2)
    assert segments == ["a b", "c d", "e"]
