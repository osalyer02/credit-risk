from __future__ import annotations

import pytest
from pydantic import ValidationError

from credit_risk.schemas.payloads import ApplicantInput


def test_applicant_schema_validates_required_fields():
    applicant = ApplicantInput(
        application_id="app-1",
        annual_income=85000,
        loan_amount=12000,
        dti=18.7,
        fico_range_low=680,
        fico_range_high=684,
        revolving_utilization=42.1,
        open_accounts=8,
        delinquencies_2y=0,
    )

    assert applicant.application_id == "app-1"


def test_applicant_schema_rejects_inverted_fico_range():
    with pytest.raises(ValidationError):
        ApplicantInput(
            application_id="app-1",
            annual_income=85000,
            loan_amount=12000,
            dti=18.7,
            fico_range_low=700,
            fico_range_high=680,
            revolving_utilization=42.1,
            open_accounts=8,
            delinquencies_2y=0,
        )
