"""Synthetic long-schema extraction-style dataset (REPORT.md 5.2 "自制长
schema 集"): 20-35 required fields per schema, long snake_case keys, enums,
nested objects and arrays. Deterministic for a given seed."""

import json
import random

S = {"type": "string"}
I = {"type": "integer"}
N = {"type": "number"}
B = {"type": "boolean"}


def enum(*values):
    return {"type": "string", "enum": list(values)}


def obj(**props):
    return {"type": "object", "properties": props, "required": list(props),
            "additionalProperties": False}


def arr(items):
    return {"type": "array", "items": items}


ADDRESS = obj(street_address_line_one=S, city_or_locality_name=S,
              state_or_province_code=S, postal_or_zip_code=S,
              country_iso_alpha2_code=S)

FIELD_POOL = {
    "record_identifier_uuid": S,
    "primary_contact_full_legal_name": S,
    "primary_contact_email_address": S,
    "primary_contact_mobile_phone_number": S,
    "date_of_birth_in_iso8601_format": S,
    "preferred_communication_channel": enum("email", "sms", "phone_call",
                                            "postal_mail"),
    "preferred_language_iso639_code": S,
    "residential_mailing_address": ADDRESS,
    "is_mailing_address_same_as_billing_address": B,
    "employment_status_category": enum("full_time_employed", "part_time_employed",
                                       "self_employed", "unemployed", "retired",
                                       "student"),
    "current_employer_organization_name": S,
    "current_job_title_or_position": S,
    "annual_gross_income_in_usd": N,
    "number_of_years_at_current_employer": I,
    "emergency_contact_person": obj(full_name=S, relationship_to_applicant=enum(
        "spouse", "parent", "sibling", "child", "friend", "other"), phone_number=S),
    "loyalty_program_membership_tier": enum("none", "bronze", "silver", "gold",
                                            "platinum"),
    "total_lifetime_purchase_amount_in_usd": N,
    "number_of_completed_orders_last_year": I,
    "has_opted_in_to_marketing_communications": B,
    "recently_viewed_product_category_names": arr(S),
    "outstanding_customer_support_ticket_count": I,
    "account_risk_assessment_level": enum("low", "medium", "high", "critical"),
    "insurance_policy_number": S,
    "insurance_coverage_plan_type": enum("basic", "standard", "premium",
                                         "comprehensive"),
    "annual_insurance_premium_amount_in_usd": N,
    "insurance_claim_incident_description": S,
    "insurance_claim_incident_date": S,
    "is_claim_under_fraud_investigation": B,
    "primary_care_physician_full_name": S,
    "known_medication_allergy_names": arr(S),
    "patient_blood_type_category": enum("a_positive", "a_negative", "b_positive",
                                        "b_negative", "ab_positive", "ab_negative",
                                        "o_positive", "o_negative"),
    "hospital_admission_reason_summary": S,
    "vehicle_identification_number": S,
    "vehicle_manufacturer_and_model_name": S,
    "vehicle_model_year": I,
    "vehicle_fuel_type_category": enum("gasoline", "diesel", "hybrid", "electric",
                                       "hydrogen"),
    "shipment_tracking_number": S,
    "shipment_carrier_service_level": enum("standard_ground", "two_day_air",
                                           "next_day_air", "international_economy"),
    "package_total_weight_in_kilograms": N,
    "list_of_shipped_items": arr(obj(item_stock_keeping_unit=S, item_quantity=I,
                                     item_unit_price_in_usd=N)),
    "real_estate_property_listing_identifier": S,
    "property_type_category": enum("single_family_home", "condominium",
                                   "townhouse", "multi_family", "vacant_land"),
    "property_total_floor_area_in_square_meters": N,
    "number_of_bedrooms_in_property": I,
    "property_listing_asking_price_in_usd": N,
    "applicant_highest_education_level": enum("high_school", "associate",
                                              "bachelor", "master", "doctorate"),
    "applicant_professional_certification_names": arr(S),
    "internal_reviewer_free_text_notes": S,
    "record_last_updated_timestamp_utc": S,
}
DOMAINS = ["customer onboarding", "insurance claim", "hospital admission",
           "vehicle registration", "logistics shipment", "real estate listing",
           "job application", "loan application"]


def make_samples(num=64, seed=1234, min_fields=20, max_fields=35):
    rng = random.Random(seed)
    keys = list(FIELD_POOL)
    out = []
    for i in range(num):
        chosen = sorted(rng.sample(range(len(keys)),
                                   rng.randint(min_fields, max_fields)))
        schema = obj(**{keys[j]: FIELD_POOL[keys[j]] for j in chosen})
        domain = rng.choice(DOMAINS)
        prompt = (
            f"Create one realistic, fictional {domain} record as JSON that "
            "fills in every field of the following JSON schema. Keep free-text "
            f"values short.\n\nSchema:\n{json.dumps(schema)}"
        )
        out.append((f"longschema-{i}", [{"role": "user", "content": prompt}],
                    schema))
    return out
