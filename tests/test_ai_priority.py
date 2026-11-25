import pytest

from ai_priority import is_ai_llm_vacancy


@pytest.mark.parametrize(
    ("title", "description"),
    [
        ("Системный аналитик (AI/LLM-платформа)", ""),
        ("Системный аналитик", "Проектирование RAG и агентных систем"),
        ("Системный аналитик", "Работа с нейросетями и промптами"),
        ("Менеджер проекта", "Поможет внедрить ИИ в процессы поддержки"),
        ("Продуктовый менеджер", "Развитие ИИ-ассистента для клиентов"),
        ("Руководитель", "Внедрение AI-инструментов в продажи"),
        ("Аналитик", "Интеграция YandexGPT и GigaChat в продукт"),
        ("Специалист", "Построение чат-ботов на базе языковых моделей"),
        ("Инженер", "Дообучение моделей и работа с эмбеддингами"),
        ("Разработчик", "Computer vision для промышленности"),
    ],
)
def test_ai_vacancies_are_detected(title: str, description: str) -> None:
    assert is_ai_llm_vacancy(title, description) is True


@pytest.mark.parametrize(
    ("title", "description"),
    [
        ("Системный аналитик", "REST API, PostgreSQL, Kafka, BPMN"),
        ("Frontend разработчик", "HTML, CSS и JavaScript"),
        ("Бухгалтер - экономист", "Ведение учёта, отчётность"),
        ("Менеджер по продажам", "Работа с B2B-клиентами"),
    ],
)
def test_plain_vacancies_are_not_flagged(title: str, description: str) -> None:
    assert is_ai_llm_vacancy(title, description) is False


def test_detection_accepts_a_single_combined_string() -> None:
    assert is_ai_llm_vacancy("Аналитик искусственного интеллекта") is True
