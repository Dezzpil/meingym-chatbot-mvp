from pydantic import BaseModel, Field
from typing import List, Optional


class Exercise(BaseModel):
    """Information about an exercise."""
    name: str = Field(None, description="Name of the exercise")
    muscles: Optional[List[str]] = Field(None, description="List of muscles that are used in the exercise")
    comments: Optional[str] = Field(None, description="Description of the exercise or comments about exercise or short instructions")


class Training(BaseModel):
    """Information about a group of exercises that are part of a training."""
    title: Optional[str] = Field(None, description="Title of the training")
    description: Optional[str] = Field(None, description="Resume, short description or comments for the training")
    exercises: Optional[List[Exercise]] = Field(None, description="List of exercises that are part of the training")


class Result(BaseModel):
    """Результат обобщения различных тренировок для пользователя"""
    comments: Optional[str] = Field(None, description="Комментарии к выбранным упражнениям и тренировке в целом")
    muscles: Optional[str] = Field(None, description="Мышцы, задейстоваванные в тренировке")
    exercises: Optional[List[str]] = Field(None, description="Список упражнений в тренировке для пользователя")