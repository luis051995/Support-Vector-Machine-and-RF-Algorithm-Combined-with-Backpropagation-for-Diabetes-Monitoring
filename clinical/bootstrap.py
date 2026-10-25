"""Composition root: conecta adaptadores con puertos."""
from clinical.application.services import ClinicalService
from clinical.adapters.persistence.repository import DjangoClinicalRepository
from clinical.adapters.ml.risk import SklearnRiskAdapter
from django.utils import timezone
def service():return ClinicalService(DjangoClinicalRepository(),SklearnRiskAdapter(),today=timezone.localdate)
