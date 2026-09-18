from pathlib import Path
import warnings

import joblib
import pandas as pd
from flask import Flask, flash, render_template, request
from sklearn.exceptions import InconsistentVersionWarning

warnings.filterwarnings("ignore", category=InconsistentVersionWarning)

app = Flask(__name__)
app.config["SECRET_KEY"] = "enterprise-revenue-prediction-secret"

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "revenue_random_forest.pkl"
FEATURE_COLUMNS_PATH = BASE_DIR / "feature_columns.pkl"

MODEL = joblib.load(MODEL_PATH)
FEATURE_COLUMNS = [str(column) for column in joblib.load(FEATURE_COLUMNS_PATH)]


def build_select_options(prefix: str):
    values = sorted(
        {
            column.replace(f"{prefix}_", "")
            for column in FEATURE_COLUMNS
            if column.startswith(f"{prefix}_")
        }
    )
    return values


JURISDICTIONS = build_select_options("enterprise_group_jurisdiction")
REVENUE_CURRENCIES = build_select_options("revenue_currency")
NET_INCOME_CURRENCIES = build_select_options("net_income_currency")
MARKET_CAP_CURRENCIES = build_select_options("market_cap_currency")
COMPANY_TYPES = build_select_options("company_type")
ENTITY_STATUS_OPTIONS = [
    "Standalone",
    "Subsidiary",
    "Holding",
    "Branch",
    "Joint Venture",
    "Other",
]


def safe_float(value, field_label):
    if value is None or str(value).strip() == "":
        raise ValueError(f"{field_label} is required.")
    try:
        return float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{field_label} must be a valid numeric value.")


def build_model_input(form_data):
    feature_row = {column: 0 for column in FEATURE_COLUMNS}

    numeric_fields = {
        "employees": "employees",
        "market_cap": "market_cap",
        "net_income": "net_income",
        "legal_units_count": "legal_units_count",
        "direct_subsidiaries_count": "direct_subsidiaries_count",
        "max_hierarchy_depth": "max_hierarchy_depth",
    }

    labels = {
        "employees": "Number of employees",
        "market_cap": "Market capitalization",
        "net_income": "Net income",
        "legal_units_count": "Legal units count",
        "direct_subsidiaries_count": "Direct subsidiaries count",
        "max_hierarchy_depth": "Maximum hierarchy depth",
    }

    for field_name, model_column in numeric_fields.items():
        feature_row[model_column] = safe_float(form_data.get(field_name), labels[field_name])

    feature_row["employees_year"] = 0.0
    feature_row["revenue_year"] = 0.0
    feature_row["net_income_year"] = 0.0

    if form_data.get("is_standalone_entity") == "True":
        feature_row["is_standalone_entity_True"] = 1.0
    else:
        feature_row["is_standalone_entity_True"] = 0.0

    category_map = {
        "enterprise_group_jurisdiction": "enterprise_group_jurisdiction",
        "company_type": "company_type",
        "revenue_currency": "revenue_currency",
        "net_income_currency": "net_income_currency",
        "market_cap_currency": "market_cap_currency",
    }

    for form_field, prefix in category_map.items():
        selected_value = (form_data.get(form_field) or "").strip()
        if not selected_value:
            raise ValueError(f"{form_field.replace('_', ' ').title()} is required.")

        model_column = f"{prefix}_{selected_value}"
        if model_column not in FEATURE_COLUMNS:
            raise ValueError(
                f"The selected {form_field.replace('_', ' ').title()} is not supported by the trained model."
            )
        feature_row[model_column] = 1.0

    if not form_data.get("enterprise_group_jurisdiction"):
        raise ValueError("Enterprise group jurisdiction is required.")

    if not form_data.get("company_type"):
        raise ValueError("Company type is required.")

    return pd.DataFrame([feature_row], columns=FEATURE_COLUMNS)


@app.route("/")
def home():
    return render_template("home.html")


@app.route("/about")
def about():
    return render_template("about.html")


@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "POST":
        try:
            form_data = request.form.to_dict(flat=True)
            input_df = build_model_input(form_data)
            prediction = float(MODEL.predict(input_df)[0])

            summary = {
                "employees": form_data.get("employees", ""),
                "market_cap": form_data.get("market_cap", ""),
                "net_income": form_data.get("net_income", ""),
                "jurisdiction": form_data.get("enterprise_group_jurisdiction", ""),
                "entity_status": form_data.get("enterprise_group_entity_status", ""),
                "company_type": form_data.get("company_type", ""),
                "revenue_currency": form_data.get("revenue_currency", ""),
                "net_income_currency": form_data.get("net_income_currency", ""),
                "market_cap_currency": form_data.get("market_cap_currency", ""),
                "legal_units_count": form_data.get("legal_units_count", ""),
                "direct_subsidiaries_count": form_data.get("direct_subsidiaries_count", ""),
                "max_hierarchy_depth": form_data.get("max_hierarchy_depth", ""),
                "is_standalone_entity": form_data.get("is_standalone_entity", ""),
            }

            if prediction < 0:
                prediction = 0.0

            return render_template(
                "result.html",
                prediction=prediction,
                summary=summary,
            )
        except ValueError as exc:
            flash(str(exc), "error")
        except Exception as exc:
            flash("Prediction could not be completed. Please review the form and try again.", "error")
            print(f"Prediction error: {exc}")

    return render_template(
        "predict.html",
        jurisdictions=JURISDICTIONS,
        company_types=COMPANY_TYPES,
        revenue_currencies=REVENUE_CURRENCIES,
        net_income_currencies=NET_INCOME_CURRENCIES,
        market_cap_currencies=MARKET_CAP_CURRENCIES,
        entity_status_options=ENTITY_STATUS_OPTIONS,
    )


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
