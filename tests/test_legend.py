import json
from pathlib import Path

from legend import Legend, load_legend, render_legend, salary_digits, salary_from_profile


def test_load_legend_renders_known_sections_and_skips_tracker(
    tmp_path: Path,
) -> None:
    path = tmp_path / "legend.json"
    path.write_text(
        json.dumps(
            {
                "general_info": {"profile": {"ФИО": "Иван Петров"}},
                "project_1_dla": {"Проект": "Конфигуратор"},
                "interview_tracker": {"private": "skip me"},
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    legend = load_legend(path)

    assert legend.available is True
    assert "Иван Петров" in legend.text
    assert "Конфигуратор" in legend.text
    assert "skip me" not in legend.text


def test_missing_or_broken_legend_is_safe(tmp_path: Path) -> None:
    missing = load_legend(tmp_path / "missing.json")
    broken_path = tmp_path / "broken.json"
    broken_path.write_text("{not json", encoding="utf-8")
    broken = load_legend(broken_path)

    assert missing == Legend()
    assert broken == Legend()
    assert load_legend(None) == Legend()


def test_salary_is_rendered_from_profile_and_digits_extracted() -> None:
    assert salary_from_profile("от 250 000 руб. на руки") == "от 250 000 рублей на руки"
    assert salary_from_profile("300000") == "от 300 000 рублей на руки"
    assert salary_from_profile("") == ""
    assert salary_digits("от 250 000 руб. на руки") == "250000"
    assert salary_digits("") == "250000"


def test_render_legend_handles_non_mapping_sections() -> None:
    assert render_legend({"general_info": "not a dict"}) == (
        '### general_info\n"not a dict"'
    )
