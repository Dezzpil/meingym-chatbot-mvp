#!/usr/bin/env python3
"""
Консольная утилита для запуска фитнес-тренера с поддержкой интерактивного режима.

Использование:
    python -m demo.main [--interactive]
    python -m demo.main "Придумай тренировку для ног"
"""

import sys
import argparse
from demo.app import App


def interactive_mode():
    """Запускает интерактивный режим работы с приложением."""
    app = App()
    print("Ассистент запущен в интерактивном режиме.")
    print("Введите ваш запрос или 'выход' для завершения.")
    
    while True:
        try:
            query = input("\nВаш запрос: ")
            if query.lower() in ['выход', 'exit', 'quit', 'q']:
                print("Завершение работы...")
                break
                
            if not query.strip():
                print("Запрос не может быть пустым. Попробуйте еще раз.")
                continue
                
            app.case1(query)
            
        except KeyboardInterrupt:
            print("\nЗавершение работы...")
            break
        except Exception as e:
            print(f"Произошла ошибка: {str(e)}")
            print("Попробуйте другой запрос или введите 'выход' для завершения.")


def single_query_mode(query):
    """Выполняет один запрос и завершает работу."""
    app = App()
    try:
        app.case1(query)
    except Exception as e:
        print(f"Произошла ошибка: {str(e)}")
        sys.exit(1)


def main():
    """Основная функция для обработки аргументов командной строки."""
    parser = argparse.ArgumentParser(
        description="Консольная утилита для запуска ассистента"
    )
    
    # Создаем группу взаимоисключающих аргументов
    group = parser.add_mutually_exclusive_group()
    
    # Аргумент для интерактивного режима
    group.add_argument(
        "-i", "--interactive",
        action="store_true",
        help="Запустить в интерактивном режиме"
    )
    
    # Аргумент для одиночного запроса
    group.add_argument(
        "query",
        nargs="?",
        help="Запрос для ассистента"
    )
    
    args = parser.parse_args()
    
    # Если указан флаг --interactive или не указан ни один аргумент, запускаем интерактивный режим
    if args.interactive or (not args.query and len(sys.argv) == 1):
        interactive_mode()
    # Иначе выполняем одиночный запрос
    elif args.query:
        single_query_mode(args.query)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()