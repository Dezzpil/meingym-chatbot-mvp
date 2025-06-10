from typing import List, Optional
from pydantic import BaseModel, Field

template1 = """
Ты - опытный действующий спортивный тренер, который очень хочет помочь мне стать сильным, здоровым и красивым.
"""

template2 = """
Я мужчина 30 лет, 175 см рост, 180 кг вес.
У меня есть физические особенности: "лишний вес, панкреатит, остеохондроз, грыжа в поясничном отделе".
Я в детстве играл в футбол, катаюсь на велосипеде.
Я начал заниматься 1 месяц и 5 дней назад и за это время выполнил 10 тренировок.
Я вернулся к активным занятиям 5 дней назад.
Моя глобальная цель тренировок на текущий момент - развитие силы (strength).
Я предпочитаю заниматься по формату сплит (split).

Я уже выполнил 2 тренировки в рамках текущего тренировочного плана:
* 1 день назад выполнил 5 упражнений, в основном на мышцы рук и немного на мышцы поясницы, плеч и ног.
* 3 дня назад выполнил 7 упражнений: в основном на мышцы ног и немного на мышцы спины.
"""

template3 = """
Проанализируй запрос пользователя, учитывая данные пользователя

Сгенерируй 2 поисковых запроса для сбора данных в интернете.
Напиши только поисковые запросы, каждый запрос на новой строке.
Поисковые запросы:
"""

class Exercise(BaseModel):
    """Information about an exercise."""
    name: str = Field(None, description="Name of the exercise")
    description: Optional[str] = Field(None, description="Description of the exercise")
    muscles: Optional[List[str]] = Field(None, description="List of muscles that are used in the exercise")
    difficulty: Optional[int] = Field(None, description="Difficulty level of the exercise")
    strength: Optional[int] = Field(None, description="Strength level of the exercise")

class Training(BaseModel):
    """Information about a group of exercises that are part of a training."""
    muscles_groups: Optional[str] = Field(None, description="Target muscle groups of the training")
    comments: Optional[str] = Field(None, description="Comments for the training")
    limitation: Optional[str] = Field(None, description="Limitation of the training")
    exercises: Optional[List[Exercise]] = Field(None, description="List of exercises that are part of the training")
    strength: Optional[int] = Field(None, description="Strength level of the training")

