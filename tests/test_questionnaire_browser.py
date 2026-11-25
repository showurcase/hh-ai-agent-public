import pytest

from questionnaire import JS_MARK_FORM_FIELDS, parse_spec

playwright = pytest.importorskip("playwright.sync_api")


TASK_BODY_HTML = """
<html><body>
  <div data-qa="task-body">
    <div data-qa="task-question">На каких проектах вы работали?</div>
    <textarea name="task_1"></textarea>
  </div>
  <div data-qa="task-body">
    <div data-qa="task-question">Готовы к переезду?</div>
    <label><input type="radio" name="task_2" value="no"> Нет</label>
    <label><input type="radio" name="task_2" value="yes"> Да</label>
  </div>
  <div data-qa="task-body">
    <div data-qa="task-question">Какие форматы подходят?</div>
    <label><input type="checkbox" name="task_3" value="remote" checked> Удалёнка</label>
    <label><input type="checkbox" name="task_3" value="office"> Офис</label>
  </div>
  <div data-qa="task-body">
    <div data-qa="task-question">Приложите тестовое задание</div>
    <input type="file" name="task_4">
  </div>
</body></html>
"""

FALLBACK_HTML = """
<html><body>
  <div>
    <label>Расскажите о себе
      <textarea name="task_1"></textarea>
    </label>
  </div>
  <div>
    <span>Готовы к переезду?</span>
    <label><input type="radio" name="task_2" value="no"> Нет</label>
    <label><input type="radio" name="task_2" value="yes"> Да</label>
  </div>
  <div>
    <select name="task_3">
      <option>Удалёнка</option>
      <option>Офис</option>
    </select>
  </div>
</body></html>
"""


@pytest.fixture(scope="module")
def browser():
    with playwright.sync_playwright() as instance:
        browser = instance.chromium.launch(
            headless=True, args=["--no-sandbox", "--disable-dev-shm-usage"]
        )
        yield browser
        browser.close()


def test_mark_form_fields_reads_task_body_blocks(browser) -> None:
    page = browser.new_page()
    try:
        page.set_content(TASK_BODY_HTML)
        spec = parse_spec(page.evaluate(JS_MARK_FORM_FIELDS))
    finally:
        page.close()

    assert [field.kind for field in spec.fields] == [
        "text",
        "single",
        "multi",
        "unknown",
    ]
    assert spec.fields[0].question == "На каких проектах вы работали?"
    assert spec.fields[1].options == ("Нет", "Да")
    assert spec.fields[2].options == ("Удалёнка", "Офис")
    assert spec.fields[3].has_file is True
    assert spec.has_file_upload() is True


def test_mark_form_fields_falls_back_to_named_task_inputs(browser) -> None:
    page = browser.new_page()
    try:
        page.set_content(FALLBACK_HTML)
        spec = parse_spec(page.evaluate(JS_MARK_FORM_FIELDS))
    finally:
        page.close()

    assert [field.kind for field in spec.fields] == ["text", "single", "select"]
    assert "Расскажите о себе" in spec.fields[0].question
    assert spec.fields[1].options == ("Нет", "Да")
    assert spec.fields[2].options == ("Удалёнка", "Офис")
