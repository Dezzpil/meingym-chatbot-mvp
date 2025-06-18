"""
Утилиты для красочного логирования в консоли.
"""

import sys
from datetime import datetime


class ColorfulLogger:
    """Класс для красочного логирования в консоли."""

    # ANSI color codes
    COLORS = {
        'RESET': '\033[0m',
        'BLACK': '\033[30m',
        'RED': '\033[31m',
        'GREEN': '\033[32m',
        'YELLOW': '\033[33m',
        'BLUE': '\033[34m',
        'MAGENTA': '\033[35m',
        'CYAN': '\033[36m',
        'WHITE': '\033[37m',
        'BOLD': '\033[1m',
        'UNDERLINE': '\033[4m',
        'BG_GREEN': '\033[42m',
        'BG_YELLOW': '\033[43m',
        'BG_BLUE': '\033[44m',
        'BG_MAGENTA': '\033[45m',
        'BG_CYAN': '\033[46m',
    }

    @staticmethod
    def _get_timestamp():
        """Возвращает текущую метку времени в формате [ЧЧ:ММ:СС]."""
        return f"[{datetime.now().strftime('%H:%M:%S')}]"

    @classmethod
    def info(cls, message, end='\n'):
        """Выводит информационное сообщение голубым цветом."""
        timestamp = cls._get_timestamp()
        print(f"{cls.COLORS['CYAN']}{timestamp} ℹ️ {message}{cls.COLORS['RESET']}", end=end, flush=True)

    @classmethod
    def success(cls, message, end='\n'):
        """Выводит сообщение об успехе зеленым цветом."""
        timestamp = cls._get_timestamp()
        print(f"{cls.COLORS['GREEN']}{timestamp} ✅ {message}{cls.COLORS['RESET']}", end=end, flush=True)

    @classmethod
    def warning(cls, message, end='\n'):
        """Выводит предупреждение желтым цветом."""
        timestamp = cls._get_timestamp()
        print(f"{cls.COLORS['YELLOW']}{timestamp} ⚠️ {message}{cls.COLORS['RESET']}", end=end, flush=True)

    @classmethod
    def error(cls, message, end='\n'):
        """Выводит сообщение об ошибке красным цветом."""
        timestamp = cls._get_timestamp()
        print(f"{cls.COLORS['RED']}{timestamp} ❌ {message}{cls.COLORS['RESET']}", end=end, flush=True)

    @classmethod
    def highlight(cls, message, end='\n'):
        """Выводит выделенное сообщение пурпурным цветом."""
        timestamp = cls._get_timestamp()
        print(f"{cls.COLORS['MAGENTA']}{timestamp} 🔍 {message}{cls.COLORS['RESET']}", end=end, flush=True)

    @classmethod
    def processing(cls, message, end='\n'):
        """Выводит сообщение о процессе синим цветом."""
        timestamp = cls._get_timestamp()
        print(f"{cls.COLORS['BLUE']}{timestamp} ⚙️ {message}{cls.COLORS['RESET']}", end=end, flush=True)

    @classmethod
    def progress(cls, current, total, prefix='', suffix='', length=30):
        """Выводит индикатор прогресса."""
        timestamp = cls._get_timestamp()
        percent = int(100 * (current / float(total)))
        filled_length = int(length * current // total)
        bar = '█' * filled_length + '░' * (length - filled_length)

        sys.stdout.write(f"\r{cls.COLORS['CYAN']}{timestamp} {prefix} |{cls.COLORS['GREEN']}{bar}{cls.COLORS['CYAN']}| {percent}% {suffix}{cls.COLORS['RESET']}")
        sys.stdout.flush()

        if current == total:
            print()


# Создаем глобальный экземпляр логгера для удобства использования
logger = ColorfulLogger()
