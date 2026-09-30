"""Stop-gradient translation-teacher constraint."""

from .translation_teacher import (
    DEFAULT_TEACHER_SHIFTS,
    TEACHER_METRICS,
    AlignedTeacher,
    TranslationTeacherKLLoss,
)

__all__ = [
    "DEFAULT_TEACHER_SHIFTS",
    "TEACHER_METRICS",
    "AlignedTeacher",
    "TranslationTeacherKLLoss",
]
