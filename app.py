
# IMPORTS
import dash
from dash import html, dcc, Input, Output, State
import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
import plotly.express as px
import joblib


# LOAD MODEL + FEATURES
model  = joblib.load("artifacts/xgboost_model.pkl")
scaler = joblib.load("artifacts/scaler.pkl")
le     = joblib.load("artifacts/label_encoder.pkl")

MODEL_FEATURES = (
    pd.read_csv("artifacts/feature_names.csv")["feature"]
    .tolist()
)

FEATURE_MEANS = pd.Series(scaler.mean_, index=MODEL_FEATURES)

# Feature importances from XGBoost (replaces SHAP)
FEATURE_IMPORTANCES = pd.Series(model.feature_importances_, index=MODEL_FEATURES)


# RISK CONFIG — colour + description per class
RISK_CONFIG = {
    "No Diabetes":  {"color": "#28a745", "bg": "#d4edda", "icon": "✅", "desc": "No indicators of diabetes detected."},
    "Pre-Diabetes": {"color": "#fd7e14", "bg": "#fff3cd", "icon": "⚠️", "desc": "Blood sugar is elevated. Lifestyle changes are recommended."},
    "Gestational":  {"color": "#6f42c1", "bg": "#e8d5f5", "icon": "🤰", "desc": "Gestational diabetes pattern detected. Note: limited training data for this class — treat with caution."},
    "Type 1":       {"color": "#dc3545", "bg": "#f8d7da", "icon": "🔴", "desc": "Type 1 diabetes indicators detected. Note: limited training data for this class — treat with caution."},
    "Type 2":       {"color": "#c82333", "bg": "#f8d7da", "icon": "🔴", "desc": "Type 2 diabetes indicators detected. Medical review advised."},
}

NAME_MAP = {
    "Column1": "Age", "bmi": "BMI", "glucose_fasting": "Fasting Glucose",
    "hba1c": "HbA1c", "systolic_bp": "Systolic BP",
    "physical_activity_minutes_per_week": "Physical Activity",
    "alcohol_consumption_per_week": "Alcohol Consumption",
    "sleep_hours_per_day": "Sleep Hours", "diabetes_risk_score": "Risk Score",
    "glucose_postprandial": "Postprandial Glucose", "insulin_level": "Insulin Level",
    "waist_to_hip_ratio": "Waist-Hip Ratio",
}


def labeled_input(label, input_component):
    return html.Div([
        html.Label(label, style={"fontSize": "0.8rem", "fontWeight": "600",
                                 "color": "#6c757d", "marginBottom": "4px",
                                 "textTransform": "uppercase", "letterSpacing": "0.05em"}),
        input_component
    ], style={"marginBottom": "12px"})


CARD = {
    "background": "white",
    "padding": "24px 28px",
    "borderRadius": "16px",
    "boxShadow": "0 2px 12px rgba(0,0,0,0.07)",
    "marginBottom": "24px",
    "border": "1px solid #f0f0f0"
}

SECTION_TITLE = {
    "fontSize": "0.7rem",
    "fontWeight": "700",
    "color": "#adb5bd",
    "textTransform": "uppercase",
    "letterSpacing": "0.1em",
    "marginBottom": "16px"
}


# DASH APP
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.FLATLY],
    suppress_callback_exceptions=True
)
server = app.server


# LAYOUT
app.layout = html.Div(style={"backgroundColor": "#f7f8fc", "minHeight": "100vh", "fontFamily": "'Segoe UI', sans-serif"}, children=[

    # NAVBAR
    html.Div(style={
        "background": "linear-gradient(135deg, #1a73e8 0%, #0d47a1 100%)",
        "padding": "18px 40px",
        "display": "flex",
        "alignItems": "center",
        "gap": "14px",
        "boxShadow": "0 2px 8px rgba(0,0,0,0.15)"
    }, children=[
        html.Span("🩺", style={"fontSize": "1.8rem"}),
        html.Div([
            html.H4("Diabetes Risk Decision Support", style={"margin": 0, "color": "white", "fontWeight": "700"}),
            html.P("BC Analytics · XGBoost Classifier",
                   style={"margin": 0, "color": "rgba(255,255,255,0.7)", "fontSize": "0.8rem"})
        ])
    ]),

    # MAIN CONTENT
    dbc.Container(fluid=False, style={"maxWidth": "960px", "paddingTop": "32px", "paddingBottom": "48px"}, children=[

        # DEMOGRAPHICS
        html.Div(style=CARD, children=[
            html.P("Patient Demographics", style=SECTION_TITLE),
            dbc.Row([
                dbc.Col(labeled_input("Age", dbc.Input(id="age", type="number", placeholder="e.g. 45", min=1, max=120)), md=4),
                dbc.Col(labeled_input("Gender", dcc.Dropdown(
                    id="gender",
                    options=[{"label": "Female", "value": "Female"},
                             {"label": "Male",   "value": "Male"},
                             {"label": "Other",  "value": "Other"}],
                    placeholder="Select gender", clearable=False
                )), md=4),
                dbc.Col(labeled_input("Ethnicity", dcc.Dropdown(
                    id="ethnicity",
                    options=[{"label": "Asian",    "value": "Asian"},
                             {"label": "Black",    "value": "Black"},
                             {"label": "Hispanic", "value": "Hispanic"},
                             {"label": "White",    "value": "White"},
                             {"label": "Other",    "value": "Other"}],
                    placeholder="Select ethnicity", clearable=False
                )), md=4),
            ])
        ]),

        # LIFESTYLE
        html.Div(style=CARD, children=[
            html.P("Lifestyle Factors", style=SECTION_TITLE),
            dbc.Row([
                dbc.Col(labeled_input("Physical Activity (min/week)",
                    dbc.Input(id="activity", type="number", placeholder="e.g. 150")), md=4),
                dbc.Col(labeled_input("Alcohol (drinks/week)",
                    dbc.Input(id="alcohol", type="number", placeholder="e.g. 2")), md=4),
                dbc.Col(labeled_input("Sleep (hours/day)",
                    dbc.Input(id="sleep", type="number", placeholder="e.g. 7")), md=4),
            ])
        ]),

        # MEDICAL
        html.Div(style=CARD, children=[
            html.P("Medical & Lab Measurements", style=SECTION_TITLE),
            dbc.Row([
                dbc.Col(labeled_input("BMI *",
                    dbc.Input(id="bmi", type="number", placeholder="e.g. 27.5")), md=3),
                dbc.Col(labeled_input("Fasting Glucose (mg/dL) *",
                    dbc.Input(id="glucose", type="number", placeholder="e.g. 110")), md=3),
                dbc.Col(labeled_input("HbA1c (%)",
                    dbc.Input(id="hba1c", type="number", placeholder="e.g. 6.5")), md=3),
                dbc.Col(labeled_input("Systolic BP (mmHg)",
                    dbc.Input(id="bp", type="number", placeholder="e.g. 120")), md=3),
            ]),
            html.P("* Required fields", style={"fontSize": "0.75rem", "color": "#adb5bd", "margin": 0})
        ]),

        # PREDICT BUTTON
        dbc.Button(
            "Assess Diabetes Risk",
            id="predict-btn",
            color="primary",
            size="lg",
            style={
                "width": "100%",
                "padding": "14px",
                "fontSize": "1rem",
                "fontWeight": "600",
                "borderRadius": "12px",
                "background": "linear-gradient(135deg, #1a73e8, #0d47a1)",
                "border": "none",
                "marginBottom": "28px",
                "boxShadow": "0 4px 12px rgba(26,115,232,0.35)"
            }
        ),

        # RESULT + CONFIDENCE
        html.Div(id="result-card"),

        # FEATURE IMPORTANCE CHART
        dcc.Graph(id="importance-plot", style={"marginTop": "8px"}),

        # FOOTER
        html.Hr(style={"borderColor": "#e9ecef", "marginTop": "40px"}),
        html.P(
            "This tool is for decision support only and does not constitute medical advice. "
            "Always consult a qualified healthcare professional.",
            style={"textAlign": "center", "fontSize": "0.75rem", "color": "#adb5bd"}
        )
    ])
])


@app.callback(
    Output("result-card", "children"),
    Output("importance-plot", "figure"),
    Input("predict-btn", "n_clicks"),
    State("age", "value"),
    State("gender", "value"),
    State("ethnicity", "value"),
    State("activity", "value"),
    State("alcohol", "value"),
    State("sleep", "value"),
    State("bmi", "value"),
    State("glucose", "value"),
    State("hba1c", "value"),
    State("bp", "value"),
)
def predict(n_clicks, age, gender, ethnicity,
            activity, alcohol, sleep,
            bmi, glucose, hba1c, bp):

    if not n_clicks:
        return "", {}

    if None in [age, bmi, glucose]:
        alert = dbc.Alert("Please fill in Age, BMI and Fasting Glucose before assessing.",
                          color="warning", style={"borderRadius": "12px"})
        return alert, {}

    try:
        # BUILD INPUT from training means, then override with user values
        data = FEATURE_MEANS.copy()
        data["Column1"]                            = age
        data["bmi"]                                = bmi
        data["glucose_fasting"]                    = glucose
        data["physical_activity_minutes_per_week"] = activity if activity is not None else data["physical_activity_minutes_per_week"]
        data["alcohol_consumption_per_week"]       = alcohol  if alcohol  is not None else data["alcohol_consumption_per_week"]
        data["sleep_hours_per_day"]                = sleep    if sleep    is not None else data["sleep_hours_per_day"]
        data["hba1c"]                              = hba1c    if hba1c    is not None else data["hba1c"]
        data["systolic_bp"]                        = bp       if bp       is not None else data["systolic_bp"]
        data["gender_Male"]        = int(gender == "Male")     if gender    else 0
        data["gender_Other"]       = int(gender == "Other")    if gender    else 0
        data["ethnicity_Black"]    = int(ethnicity == "Black")    if ethnicity else 0
        data["ethnicity_Hispanic"] = int(ethnicity == "Hispanic") if ethnicity else 0
        data["ethnicity_White"]    = int(ethnicity == "White")    if ethnicity else 0
        data["ethnicity_Other"]    = int(ethnicity == "Other")    if ethnicity else 0

        X_raw    = pd.DataFrame([data], columns=MODEL_FEATURES)
        X_scaled = pd.DataFrame(scaler.transform(X_raw), columns=MODEL_FEATURES)

        # PREDICT + CONFIDENCE
        prediction  = model.predict(X_scaled)[0]
        label       = le.inverse_transform([prediction])[0]
        proba       = model.predict_proba(X_scaled)[0]
        cfg         = RISK_CONFIG.get(label, {"color": "#6c757d", "bg": "#f8f9fa", "icon": "❓", "desc": ""})

        # RESULT CARD
        all_classes = le.classes_
        prob_rows = [
            dbc.Col(html.Div([
                html.Div(cls, style={"fontSize": "0.7rem", "color": "#6c757d", "marginBottom": "4px"}),
                dbc.Progress(
                    value=round(proba[i] * 100, 1),
                    label=f"{round(proba[i]*100,1)}%",
                    color="primary" if cls == label else "secondary",
                    style={"height": "10px", "borderRadius": "6px"}
                )
            ]), md=4, style={"marginBottom": "10px"})
            for i, cls in enumerate(all_classes)
        ]

        result_card = html.Div(style={
            **CARD,
            "borderLeft": f"6px solid {cfg['color']}",
            "backgroundColor": cfg["bg"]
        }, children=[
            html.Div(style={"display": "flex", "alignItems": "center", "gap": "12px", "marginBottom": "8px"}, children=[
                html.Span(cfg["icon"], style={"fontSize": "2rem"}),
                html.Div([
                    html.H4(label, style={"margin": 0, "color": cfg["color"], "fontWeight": "700"}),
                    html.P(cfg["desc"], style={"margin": 0, "color": "#495057", "fontSize": "0.9rem"})
                ])
            ]),
            html.Hr(style={"borderColor": cfg["color"], "opacity": "0.3"}),
            html.P("Class Probabilities", style={**SECTION_TITLE, "marginBottom": "10px"}),
            dbc.Row(prob_rows)
        ])

        # FEATURE IMPORTANCE (top 10)
        imp_df = (
            FEATURE_IMPORTANCES
            .sort_values(ascending=False)
            .head(10)
            .reset_index()
        )
        imp_df.columns = ["Feature", "Importance"]
        imp_df["Feature"] = imp_df["Feature"].apply(lambda x: NAME_MAP.get(x, x.replace("_", " ").title()))

        fig = px.bar(
            imp_df,
            x="Importance",
            y="Feature",
            orientation="h",
            title="Top 10 Most Important Features",
            labels={"Importance": "Feature Importance", "Feature": ""}
        )
        fig.update_traces(marker_color="#1a73e8")
        fig.update_layout(
            yaxis=dict(autorange="reversed"),
            template="plotly_white",
            height=420,
            title_font_size=14,
            margin=dict(l=10, r=10, t=50, b=10),
            paper_bgcolor="white",
            plot_bgcolor="white",
        )

        return result_card, fig

    except Exception as e:
        import traceback
        err = dbc.Alert(f"Error: {traceback.format_exc()}", color="danger", style={"borderRadius": "12px", "whiteSpace": "pre-wrap"})
        return err, {}


# RUN
if __name__ == "__main__":
    try:
        app.run(debug=True)
    except AttributeError:
        app.run_server(debug=True)
