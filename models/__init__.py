# This file makes the models directory a Python package 

from models.database import db, User, Course, Tutor, Material, StudyPlan, SelectedCourse, SelectedTutor, SelectedMaterial
from models.models import StudyPlanItem

__all__ = [
    'db',
    'User',
    'Course',
    'Tutor',
    'Material',
    'StudyPlan',
    'SelectedCourse',
    'SelectedTutor',
    'SelectedMaterial',
    'StudyPlanItem'
] 