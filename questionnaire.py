from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class QuestionnaireField:
    index: int
    kind: str
    question: str
    options: tuple[str, ...] = ()
    has_file: bool = False

    def as_payload(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "kind": self.kind,
            "question": self.question,
            "options": list(self.options),
        }


@dataclass
class QuestionnaireSpec:
    fields: list[QuestionnaireField] = field(default_factory=list)

    @property
    def is_empty(self) -> bool:
        return not self.fields

    def has_file_upload(self) -> bool:
        return any(item.has_file for item in self.fields)

    def unsupported(self) -> list[QuestionnaireField]:
        return [item for item in self.fields if item.kind == "unknown"]

    def payload(self) -> list[dict[str, Any]]:
        return [item.as_payload() for item in self.fields]


def parse_spec(raw: object) -> QuestionnaireSpec:
    if not isinstance(raw, list):
        return QuestionnaireSpec()
    fields: list[QuestionnaireField] = []
    for position, item in enumerate(raw):
        if not isinstance(item, dict):
            continue
        try:
            index = int(item.get("index", position))
        except (TypeError, ValueError):
            index = position
        kind = str(item.get("kind") or "unknown")
        if kind not in {"text", "single", "multi", "select", "unknown"}:
            kind = "unknown"
        question = str(item.get("question") or "").strip()
        options_raw = item.get("options")
        # Keep empty labels in place: the LLM's choice index must match the
        # input's data-bot-option index, even when one radio has no label.
        options = (
            tuple(str(option).strip() for option in options_raw)
            if isinstance(options_raw, list)
            else ()
        )
        if kind in {"single", "multi", "select"} and not any(options):
            kind = "unknown"
        fields.append(
            QuestionnaireField(
                index=index,
                kind=kind,
                question=question,
                options=options,
                has_file=bool(item.get("has_file")),
            )
        )
    return QuestionnaireSpec(fields)


# Marks each logical question with data-bot-field="<index>" and each choice with
# data-bot-option="<index>:<choice>" so Playwright can address them reliably even
# when HH renders several identical input names or custom switches.
JS_MARK_FORM_FIELDS = r"""
() => {
  const clean = (value) => (value || '')
    .replace(/[\u200B-\u200D\u2060\uFEFF]/g, '')
    .replace(/\s+/g, ' ')
    .trim();

  document.querySelectorAll('[data-bot-field]').forEach((node) => {
    node.removeAttribute('data-bot-field');
  });
  document.querySelectorAll('[data-bot-option]').forEach((node) => {
    node.removeAttribute('data-bot-option');
  });

  const optionLabel = (input, scope) => {
    const label = input.closest('label')
      || (input.id && scope.querySelector('label[for="' + CSS.escape(input.id) + '"]'));
    return clean((label && label.innerText) || input.getAttribute('aria-label') || input.value);
  };

  const questionText = (container) => {
    const explicit = container.querySelector(
      '[data-qa="task-question"], [data-qa="task-title"], [data-qa*="question-title"], legend'
    );
    if (explicit) {
      return clean(explicit.innerText);
    }
    const clone = container.cloneNode(true);
    clone.querySelectorAll('input, textarea, select, button, script, style').forEach((node) => {
      const label = node.closest('label');
      if (label && label.querySelectorAll('input, textarea, select').length <= 1) {
        label.remove();
      } else {
        node.remove();
      }
    });
    let text = clean(clone.innerText);
    if (text.length > 800) {
      text = text.slice(0, 800);
    }
    return text;
  };

  const result = [];
  const mark = (container, index, question, kind, controls, hasFile) => {
    container.setAttribute('data-bot-field', String(index));
    controls.forEach((input, choice) => {
      input.setAttribute('data-bot-option', index + ':' + choice);
    });
    result.push({
      index,
      kind,
      question,
      options: controls.map((input) => optionLabel(input, container)),
      has_file: hasFile,
    });
  };

  const blocks = Array.from(document.querySelectorAll('[data-qa="task-body"]'));
  if (blocks.length) {
    blocks.forEach((block, index) => {
      const controls = Array.from(
        block.querySelectorAll('input[type="radio"], input[type="checkbox"]')
      );
      const hasFile = !!block.querySelector('input[type="file"]');
      const select = block.querySelector('select');
      let kind = 'unknown';
      if (controls.length) {
        kind = controls[0].type === 'radio' ? 'single' : 'multi';
      } else if (select) {
        kind = 'select';
      } else if (block.querySelector('textarea, input[type="text"], input:not([type])')) {
        kind = 'text';
      }
      if (kind === 'select') {
        const options = Array.from(select.options).map((option) => clean(option.text));
        block.setAttribute('data-bot-field', String(index));
        result.push({
          index,
          kind,
          question: questionText(block),
          options,
          has_file: hasFile,
        });
        return;
      }
      mark(block, index, questionText(block), kind, controls, hasFile);
    });
    return result;
  }

  // Fallback for the inline response modal: HH puts names like task_123 on
  // textareas and radio groups without a task-body wrapper.
  const controls = Array.from(document.querySelectorAll(
    'textarea[name^="task_"], input[name^="task_"], select[name^="task_"]'
  ));
  const hasFile = !!document.querySelector('input[type="file"][name^="task_"]');
  const groups = new Map();
  const ordered = [];
  controls.forEach((input) => {
    const type = (input.getAttribute('type') || input.tagName).toLowerCase();
    if (input.tagName === 'SELECT') {
      const key = 'select:' + (input.name || input.id || ordered.length);
      groups.set(key, { type: 'select', inputs: [input] });
      ordered.push(key);
      return;
    }
    if (type === 'radio' || type === 'checkbox') {
      const key = 'group:' + (input.name || 'anonymous');
      if (!groups.has(key)) {
        groups.set(key, { type: 'group', inputs: [] });
        ordered.push(key);
      }
      groups.get(key).inputs.push(input);
    } else {
      const key = 'single:' + (input.name || input.id || ordered.length);
      groups.set(key, { type: 'single', inputs: [input] });
      ordered.push(key);
    }
  });

  ordered.forEach((key, index) => {
    const group = groups.get(key);
    const first = group.inputs[0];
    let container = first.parentElement || first;
    for (let depth = 0; container && depth < 3; depth++) {
      const text = clean(container.innerText);
      if (text.length >= 3 && text.length <= 600) {
        break;
      }
      container = container.parentElement || container;
    }
    const hint = clean(
      first.getAttribute('aria-label') || first.getAttribute('placeholder')
    );
    const fallbackQuestion = () => {
      const text = questionText(container);
      if (text) {
        return text;
      }
      return hint;
    };
    const isGroup = group.type === 'group';
    const isSelect = group.type === 'select';
    const kind = isSelect
      ? 'select'
      : isGroup
        ? (first.type === 'radio' ? 'single' : 'multi')
        : 'text';
    if (isSelect) {
      container.setAttribute('data-bot-field', String(index));
      result.push({
        index,
        kind,
        question: fallbackQuestion(),
        options: Array.from(first.options).map((option) => clean(option.text)),
        has_file: hasFile,
      });
      return;
    }
    mark(
      container,
      index,
      fallbackQuestion(),
      kind,
      isGroup ? group.inputs : [],
      hasFile
    );
    group.inputs.forEach((input, choice) => {
      input.setAttribute('data-bot-option', index + ':' + choice);
    });
  });
  return result;
}
"""
