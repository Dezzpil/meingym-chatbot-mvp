#!/usr/bin/env python3
"""
Тесты для проверки работоспособности консольной утилиты.
"""

import unittest
import sys
import os
from unittest.mock import patch, MagicMock

# Добавляем родительскую директорию в sys.path для импорта модулей
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from demo.main import main, interactive_mode, single_query_mode
from demo.app import App


class TestMain(unittest.TestCase):
    """Тесты для проверки функциональности main.py."""

    def test_main_exists(self):
        """Проверяет, что функция main существует."""
        self.assertTrue(callable(main))

    def test_interactive_mode_exists(self):
        """Проверяет, что функция interactive_mode существует."""
        self.assertTrue(callable(interactive_mode))

    def test_single_query_mode_exists(self):
        """Проверяет, что функция single_query_mode существует."""
        self.assertTrue(callable(single_query_mode))

    @patch('demo.main.App')
    @patch('demo.main.input', return_value='выход')
    @patch('builtins.print')
    def test_interactive_mode_exit(self, mock_print, mock_input, mock_app):
        """Проверяет, что интерактивный режим корректно завершается при вводе 'выход'."""
        interactive_mode()
        # Проверяем, что App был инициализирован
        mock_app.assert_called_once()
        # Проверяем, что был запрос на ввод
        mock_input.assert_called_once()
        # Проверяем, что было сообщение о завершении
        mock_print.assert_any_call("Завершение работы...")

    @patch('demo.main.App')
    @patch('builtins.print')
    def test_single_query_mode(self, mock_print, mock_app):
        """Проверяет, что одиночный режим корректно вызывает App.case1."""
        # Создаем мок для App и его метода case1
        mock_app_instance = MagicMock()
        mock_app.return_value = mock_app_instance
        
        # Вызываем функцию с тестовым запросом
        single_query_mode("тестовый запрос")
        
        # Проверяем, что App был инициализирован
        mock_app.assert_called_once()
        # Проверяем, что был вызван метод case1 с правильным аргументом
        mock_app_instance.case1.assert_called_once_with("тестовый запрос")


if __name__ == '__main__':
    unittest.main()