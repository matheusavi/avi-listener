"""Cutting the transcript at speaker changes.

Whisper chooses its own segment boundaries, routinely 10-15 seconds long, and
one segment often spans several diarization turns. Labelling whole segments
discards every speaker change inside one, and a participant whose turns are all
short disappears from the transcript completely.

The numbers below come from a real recording: diarization found turns of 8.9s,
2.0s, 11.4s, 4.4s and 2.2s, while Whisper produced segments of 7.6s, 13.3s and
6.7s. speaker_0, who only ever spoke in the two short turns, was missing.
"""

from __future__ import annotations

from avilistener.meeting import DiarizationTurn, split_words_by_speaker


def turns(*spans: tuple[float, float, str]) -> list[DiarizationTurn]:
    return [DiarizationTurn(start=s, end=e, speaker=name) for s, e, name in spans]


def words(*items: tuple[float, float, str]) -> list[tuple[float, float, str]]:
    return list(items)


class TestSplitWordsBySpeaker:
    def test_one_speaker_stays_one_line(self) -> None:
        lines = split_words_by_speaker(
            words((0.0, 1.0, "hello"), (1.0, 2.0, "there"), (2.0, 3.0, "friend")),
            turns((0.0, 5.0, "speaker_0")),
        )
        assert len(lines) == 1
        assert lines[0].speaker == "speaker_0"
        assert lines[0].text == "hello there friend"
        assert (lines[0].start, lines[0].end) == (0.0, 3.0)

    def test_splits_where_the_speaker_changes_mid_segment(self) -> None:
        lines = split_words_by_speaker(
            words(
                (0.0, 1.0, "who"), (1.0, 2.0, "is"), (2.0, 3.0, "next"),
                (5.0, 6.0, "that"), (6.0, 7.0, "is"), (7.0, 8.0, "me"),
            ),
            turns((0.0, 4.0, "speaker_0"), (4.0, 9.0, "speaker_1")),
        )
        assert [line.speaker for line in lines] == ["speaker_0", "speaker_1"]
        assert lines[0].text == "who is next"
        assert lines[1].text == "that is me"

    def test_a_short_turn_inside_a_long_segment_is_not_lost(self) -> None:
        """The regression this exists for: speaker_0 spoke only briefly."""
        lines = split_words_by_speaker(
            words(
                (0.0, 1.0, "long"), (1.0, 2.0, "explanation"), (2.0, 3.0, "continues"),
                (3.2, 4.0, "quick"), (4.0, 4.8, "interjection"),
                (5.2, 6.0, "and"), (6.0, 7.0, "back"), (7.0, 8.0, "again"),
            ),
            turns((0.0, 3.1, "speaker_1"), (3.1, 5.0, "speaker_0"), (5.0, 9.0, "speaker_1")),
        )
        assert [line.speaker for line in lines] == ["speaker_1", "speaker_0", "speaker_1"]
        assert "speaker_0" in {line.speaker for line in lines}
        assert lines[1].text == "quick interjection"

    def test_line_timestamps_follow_the_words_not_the_segment(self) -> None:
        lines = split_words_by_speaker(
            words((10.0, 11.0, "first"), (20.0, 21.0, "second")),
            turns((0.0, 15.0, "speaker_0"), (15.0, 30.0, "speaker_1")),
        )
        assert (lines[0].start, lines[0].end) == (10.0, 11.0)
        assert (lines[1].start, lines[1].end) == (20.0, 21.0)

    def test_a_stray_word_does_not_flip_the_speaker(self) -> None:
        """Diarization boundaries are approximate; one word must not split a line."""
        lines = split_words_by_speaker(
            words(
                (0.0, 1.0, "we"), (1.0, 2.0, "were"), (2.0, 3.0, "saying"),
                (3.0, 3.2, "um"),  # 0.2s, lands across a boundary
                (3.3, 4.3, "that"), (4.3, 5.3, "thing"),
            ),
            turns((0.0, 3.05, "speaker_0"), (3.05, 3.25, "speaker_1"), (3.25, 6.0, "speaker_0")),
        )
        assert [line.speaker for line in lines] == ["speaker_0"]
        assert "um" in lines[0].text

    def test_a_lone_word_never_gets_its_own_line(self) -> None:
        """Filler carries a long timestamp spanning the pause around it.

        Observed in a real transcript: a single "e" was given 1.4s and split a
        sentence in half, so a duration test alone is not enough.
        """
        lines = split_words_by_speaker(
            words(
                (0.0, 2.0, "tranquilo"), (2.0, 4.0, "eu"), (4.0, 6.0, "falo"),
                (6.0, 7.4, "e"),  # 1.4s, longer than min_line_seconds
                (7.4, 9.0, "bom"), (9.0, 11.0, "seguimos"),
            ),
            turns((0.0, 6.1, "speaker_0"), (6.1, 7.3, "speaker_1"), (7.3, 12.0, "speaker_0")),
        )
        assert [line.speaker for line in lines] == ["speaker_0"]
        assert "e" in lines[0].text.split()

    def test_a_real_short_turn_with_several_words_survives(self) -> None:
        """Smoothing must not swallow genuine brief interjections."""
        lines = split_words_by_speaker(
            words(
                (0.0, 2.0, "so"), (2.0, 4.0, "anyway"),
                (4.1, 4.6, "no"), (4.6, 5.1, "that's"), (5.1, 5.6, "wrong"),
                (6.0, 8.0, "continuing"),
            ),
            turns((0.0, 4.05, "speaker_0"), (4.05, 5.7, "speaker_1"), (5.7, 9.0, "speaker_0")),
        )
        assert [line.speaker for line in lines] == ["speaker_0", "speaker_1", "speaker_0"]
        assert lines[1].text == "no that's wrong"

    def test_blank_words_are_skipped(self) -> None:
        lines = split_words_by_speaker(
            words((0.0, 1.0, "  "), (1.0, 2.0, "hello")),
            turns((0.0, 5.0, "speaker_0")),
        )
        assert lines[0].text == "hello"

    def test_no_words_produces_no_lines(self) -> None:
        assert split_words_by_speaker([], turns((0.0, 5.0, "speaker_0"))) == []

    def test_words_outside_every_turn_still_appear(self) -> None:
        """Text must never be dropped just because diarization missed it."""
        lines = split_words_by_speaker(
            words((50.0, 51.0, "orphan")),
            turns((0.0, 5.0, "speaker_0")),
        )
        assert len(lines) == 1
        assert lines[0].text == "orphan"
