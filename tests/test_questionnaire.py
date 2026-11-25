from questionnaire import (
    JS_MARK_FORM_FIELDS,
    QuestionnaireField,
    QuestionnaireSpec,
    parse_spec,
)


def test_parse_spec_reads_text_and_choice_fields() -> None:
    spec = parse_spec(
        [
            {
                "index": 0,
                "kind": "text",
                "question": "Расскажите о проектах",
                "options": [],
                "has_file": False,
            },
            {
                "index": 1,
                "kind": "single",
                "question": "Готовы к переезду?",
                "options": ["Нет", "Да"],
            },
            {
                "index": 2,
                "kind": "multi",
                "question": "Какие форматы подходят?",
                "options": ["Удалёнка", "Гибрид", "Офис"],
            },
        ]
    )

    assert spec.fields[0] == QuestionnaireField(
        index=0, kind="text", question="Расскажите о проектах"
    )
    assert spec.fields[1].options == ("Нет", "Да")
    assert spec.payload()[1] == {
        "index": 1,
        "kind": "single",
        "question": "Готовы к переезду?",
        "options": ["Нет", "Да"],
    }


def test_parse_spec_downgrades_choice_without_options_to_unknown() -> None:
    spec = parse_spec([{"index": 0, "kind": "single", "question": "?", "options": []}])

    assert spec.fields[0].kind == "unknown"
    assert spec.unsupported() == [spec.fields[0]]


def test_parse_spec_keeps_empty_option_slots_aligned() -> None:
    spec = parse_spec(
        [{"index": 0, "kind": "single", "question": "?", "options": ["Да", "", "Нет"]}]
    )

    assert spec.fields[0].options == ("Да", "", "Нет")
    assert spec.fields[0].kind == "single"


def test_parse_spec_detects_file_upload_and_bad_payload() -> None:
    spec = parse_spec(
        [{"index": 0, "kind": "unknown", "question": "Тест", "has_file": True}]
    )

    assert spec.has_file_upload() is True
    assert parse_spec("garbage") == QuestionnaireSpec()
    assert parse_spec([None, 42]) == QuestionnaireSpec()


def test_form_js_marks_fields_and_options() -> None:
    assert 'data-bot-field' in JS_MARK_FORM_FIELDS
    assert 'data-bot-option' in JS_MARK_FORM_FIELDS
    assert 'data-qa="task-body"' in JS_MARK_FORM_FIELDS
    assert 'name^="task_"' in JS_MARK_FORM_FIELDS
