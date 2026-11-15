from archiver import fidelity

RU_RAW = [
    (0, "ну вот значит э мы мы запустили этот проект в двадцать втором году и как бы"),
    (1, "сначала никто не верил что что это вообще взлетит понимаешь потому что рынок был"),
    (2, "э совсем маленький и инвесторы говорили что такие сервисы никому не нужны"),
    (3, "но мы короче решили попробовать и за первый год набрали около десяти тысяч"),
    (4, "пользователей причём почти без рекламы только за счёт сарафанного радио"),
]
RU_CLEAN = [
    (
        0,
        (
            "Мы запустили этот проект в двадцать втором году, и сначала никто не верил, что это "
            "вообще взлетит, потому что рынок был совсем маленький и инвесторы говорили, что такие "
            "сервисы никому не нужны."
        ),
    ),
    (
        1,
        (
            "Но мы решили попробовать и за первый год набрали около десяти тысяч пользователей, "
            "причём почти без рекламы, только за счёт сарафанного радио."
        ),
    ),
]

EN_RAW = [
    (0, "so um the the main thing about cubernetes is that you know it it doesnt care"),
    (1, "where your containers run uh it just schedules them on whatever nodes have room and"),
    (2, "if a node dies it like reschedules everything somewhere else which is I mean"),
    (3, "the whole point you stop thinking about individual machines and start thinking"),
    (4, "about the cluster as one big computer that you hand work to"),
]
EN_CLEAN = [
    (
        0,
        (
            "So the main thing about Kubernetes is that it doesn't care where your containers run. "
            "It just schedules them on whatever nodes have room, and if a node dies, it reschedules "
            "everything somewhere else."
        ),
    ),
    (
        1,
        (
            "Which is the whole point: you stop thinking about individual machines and start "
            "thinking about the cluster as one big computer that you hand work to."
        ),
    ),
]


def test_tokenize_folds_case_yo_and_punctuation():
    assert fidelity.tokenize("Ещё раз, Ёжик!") == ["еще", "раз", "ежик"]


def test_fillers_and_repeats_are_ignored():
    tokens = [t.text for t in fidelity.content_tokens([(0, "ну как бы я я я думаю, you know")])]
    assert tokens == ["я", "думаю"]


def test_light_russian_cleanup_passes():
    report = fidelity.check(RU_RAW, RU_CLEAN)
    assert report.ok, report.reason
    assert report.recall > 0.9


def test_light_english_cleanup_passes():
    report = fidelity.check(EN_RAW, EN_CLEAN)
    assert report.ok, report.reason


def test_summary_fails_on_recall():
    summary = [(0, "Проект запустили в 2022 году, и он быстро набрал аудиторию без рекламы.")]
    report = fidelity.check(RU_RAW, summary)
    assert not report.ok
    assert "recall" in report.reason


def test_translation_fails():
    report = fidelity.check(RU_RAW, EN_CLEAN)
    assert not report.ok
    assert "recall" in report.reason and "novelty" in report.reason


def test_dropped_passage_fails_on_gap_even_with_good_recall():
    words = [f"слово{i}" for i in range(400)]
    source = [(i // 10, " ".join(words[i : i + 10])) for i in range(0, 400, 10)]
    kept = words[:200] + words[260:]
    report = fidelity.check(source, [(0, " ".join(kept))])
    assert report.recall >= fidelity.RECALL_MIN
    assert report.longest_gap == 60
    assert not report.ok
    assert "vanished" in report.reason


def test_empty_inputs():
    assert fidelity.check([], []).ok
    assert not fidelity.check(RU_RAW, []).ok


def test_alignment_maps_paragraph_to_source_segment():
    alignment = fidelity.align(RU_RAW, RU_CLEAN)
    assert alignment.source_ref_for(0) == 0
    # The second paragraph begins with "Но мы решили", spoken in segment 3.
    assert alignment.source_ref_for(1) == 3
    assert alignment.source_ref_for(7) is None
