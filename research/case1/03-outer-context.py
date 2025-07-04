
# надо добавить внешний контекст
# как он может выглядеть?

user_context = {
    "user": {
        "height": 175,
        "weight": 180,
        "age": 30,
        "sex": "male",
        "aggravatingFacts": "лишний вес, панкреатит, остеохондроз, грыжа в поясничном отделе",
        "experience": "в детстве играл в футбол, катаюсь на велосипеде"
    },
    "training_period": {
        "startedAt": "2025-05-24",
        "purpose": "STRENGTH",
        "type": "SPLIT",
        "completedTrainings": {
            "2025-05-20": "Руки: 6; Поясница: 2; Плечи: 2; Ноги: 1"
        }
    },
    "totalCompletedTrainings": 10,
    "firstCompletedTraining": "2025-04-10"
}

# Я сомневаюсь, что если это добавить в контекст модели - она хорошо разберет
# что здесь к чему. Лучше генерировать текстовое описание, которое можно добавить в
# в промпт. И пусть веб-аппа сразу предоставляет эти данные при начале общения с чат-ботом.
# Т.е. эти данные будут всегда доступны.

user_context_str = """
Характеристика пользователя:
* Мужчина 30 лет, 175 см рост, 180 кг вес.
* Указал следующие проблемы: "лишний вес, панкреатит, остеохондроз, грыжа в поясничном отделе"
* Описал свой спортивный опыт и физическую форму: "в детстве играл в футбол, катаюсь на велосипеде"
* Начал заниматься в приложении 1 месяц и 5 дней назад и за это время выполнил 10 тренировок.
* Вернулся к активным занятиям 5 дней назад.
* Указал развитие силы (strength) как главную цель тренировок.
* Предпочитает заниматься по формату сплит (split).

История тренировок в текущем тренировочном периоде:
Выполнил 2 тренировки в рамках активного плана.

1. Тренировка 1 день назад. 
Выполнил 5 упражнений: .... 
В рамках выполнения были задействованы мышцы разных групп в качестве мышц-агонистов: 
* Руки: 6 раз
* Поясницы: 2 раз 
* Плеч: 2 раза
* Ноги: 1 раз

2. Тренировка 3 день назад. 
Выполнил 7 упражнений: .... 
В рамках выполнения были задействованы мышцы разных групп в качестве мышц-агонистов: 
* Ноги: 6 раз
* Спина: 3 раз
"""

user_context_str_empty = """
Характеристика пользователя:
* Женщина 35 лет, 164 см рост, 80 кг вес.
* Указала следующие проблемы: "..." # можно заменить на "Не указала".
* Описала свой спортивный опыт и физическую форму: "..." # можно заменить на нет опыта
* Еще не начала заниматься в приложении.
* Указала снижение веса (weight loss) как главную цель тренировок. # default
* Предпочитает заниматься по формату фулл-бади (full-body). # default

История тренировок в текущем тренировочном периоде:
Еще не начала заниматься.
"""

## общие замечания к составлению промта:
# менять род глаголов в зависимости от пола
# менять склонение существительных в зависимости от числа
