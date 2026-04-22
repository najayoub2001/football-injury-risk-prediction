import pandas as pd
import numpy as np

MODEL_COLUMNS = [
    "Age",
    "Height_cm",
    "Weight_kg",
    "Position",
    "Training_Hours_Per_Week",
    "Matches_Played_Past_Season",
    "Previous_Injury_Count",
    "Knee_Strength_Score",
    "Hamstring_Flexibility",
    "Reaction_Time_ms",
    "Balance_Test_Score",
    "Sprint_Speed_10m_s",
    "Agility_Score",
    "Sleep_Hours_Per_Night",
    "Stress_Level_Score",
    "Nutrition_Quality_Score",
    "Warmup_Routine_Adherence",
    "BMI"
]

def build_model_input(player_dict: dict) -> pd.DataFrame:
    bmi = player_dict["weight_kg"] / ((player_dict["height_cm"] / 100) ** 2)

    return pd.DataFrame({
        "Age": [player_dict["age"]],
        "Height_cm": [player_dict["height_cm"]],
        "Weight_kg": [player_dict["weight_kg"]],
        "Position": [player_dict["position"]],
        "Training_Hours_Per_Week": [player_dict["training_hours"]],
        "Matches_Played_Past_Season": [player_dict["matches_played"]],
        "Previous_Injury_Count": [player_dict["previous_injuries"]],
        "Knee_Strength_Score": [player_dict["knee_strength"]],
        "Hamstring_Flexibility": [player_dict["hamstring_flexibility"]],
        "Reaction_Time_ms": [player_dict["reaction_time"]],
        "Balance_Test_Score": [player_dict["balance_score"]],
        "Sprint_Speed_10m_s": [player_dict["sprint_speed"]],
        "Agility_Score": [player_dict["agility"]],
        "Sleep_Hours_Per_Night": [player_dict["sleep_hours"]],
        "Stress_Level_Score": [player_dict["stress"]],
        "Nutrition_Quality_Score": [player_dict["nutrition"]],
        "Warmup_Routine_Adherence": [player_dict["warmup"]],
        "BMI": [bmi]
    })[MODEL_COLUMNS]

def risk_band(probability: float) -> str:
    if probability < 0.40:
        return "Low"
    elif probability < 0.75:
        return "Moderate"
    return "High"

def risk_colour(probability: float) -> str:
    if probability < 0.40:
        return "#22c55e"
    elif probability < 0.75:
        return "#f59e0b"
    return "#ef4444"

def likely_injury_zone(player_dict: dict) -> str:
    if player_dict["hamstring_flexibility"] <= 4 and player_dict["sprint_speed"] >= 3.0:
        return "Hamstring / Posterior Thigh"
    if player_dict["knee_strength"] <= 4:
        return "Knee"
    if player_dict["balance_score"] <= 4:
        return "Ankle / Lower Leg"
    if player_dict["stress"] >= 8 and player_dict["sleep_hours"] <= 5:
        return "General Fatigue / Whole Body"
    if player_dict["previous_injuries"] >= 3:
        return "Previously Affected Lower Limb"
    return "Lower Limb"

def zone_marker(zone_name: str):
    markers = {
        "Head": (341, 95),
        "Upper Body": (341, 250),


        "Quadriceps": (341, 610),
        "Hamstring": (315, 635),   # proxy position on upper thigh
        "Knee": (341, 755),
        "Ankle": (341, 920),

        "Lower Limb": (341, 700),
        "Previously Affected Lower Limb": (341, 720),
        "General Fatigue / Whole Body": (341, 180),
    }

    return markers.get(zone_name, markers["Lower Limb"])

def simulate_30_day_trend(player_dict: dict, model) -> pd.DataFrame:
    days = np.arange(1, 31)
    rows = []

    base_training = player_dict["training_hours"]
    base_sleep = player_dict["sleep_hours"]
    base_stress = player_dict["stress"]

    for d in days:
        sim = player_dict.copy()
        sim["training_hours"] = max(1, min(30, base_training + np.sin(d / 3) * 2 + (d % 7 == 0) * 2))
        sim["sleep_hours"] = max(3, min(10, base_sleep - (d % 6 == 0) * 1 + np.cos(d / 4) * 0.3))
        sim["stress"] = max(1, min(10, base_stress + (d % 5 == 0) * 1 + np.sin(d / 5)))
        sim["matches_played"] = player_dict["matches_played"]

        X = build_model_input(sim)
        prob = float(model.predict_proba(X)[0][1])

        rows.append({
            "Day": d,
            "Injury Risk": prob,
            "Training Hours": sim["training_hours"],
            "Sleep Hours": sim["sleep_hours"],
            "Stress": sim["stress"]
        })

    return pd.DataFrame(rows)

import random

def scale_1_to_9(value: int, min_val: float, max_val: float) -> float:
    return min_val + (value - 1) * (max_val - min_val) / 8

def descale_to_1_to_9(value: float, min_val: float, max_val: float) -> int:
    if max_val == min_val:
        return 5
    scaled = 1 + ((value - min_val) * 8 / (max_val - min_val))
    return int(round(max(1, min(9, scaled))))

def readiness_score(player_dict: dict) -> int:
    sleep_component = max(0, min(100, (player_dict["sleep_hours"] - 5) / 5 * 100))
    stress_component = max(0, min(100, 100 - ((player_dict["stress"] - 20) / 70 * 100)))
    nutrition_component = max(0, min(100, ((player_dict["nutrition"] - 60) / 30 * 100)))
    warmup_component = max(0, min(100, (player_dict["warmup"] / 10 * 100)))
    balance_component = max(0, min(100, ((player_dict["balance_score"] - 70) / 25 * 100)))

    score = (
        0.25 * sleep_component +
        0.25 * stress_component +
        0.20 * nutrition_component +
        0.15 * warmup_component +
        0.15 * balance_component
    )
    return int(round(score))

def get_risk_drivers(player_dict: dict) -> list[str]:
    drivers = []

    if player_dict["stress"] >= 70:
        drivers.append("High stress load")
    if player_dict["sleep_hours"] <= 6.0:
        drivers.append("Low sleep / poor recovery")
    if player_dict["previous_injuries"] >= 2:
        drivers.append("Repeated injury history")
    if player_dict["knee_strength"] <= 68:
        drivers.append("Low knee strength")
    if player_dict["hamstring_flexibility"] <= 68:
        drivers.append("Limited hamstring flexibility")
    if player_dict["balance_score"] <= 78:
        drivers.append("Reduced balance / stability")
    if player_dict["training_hours"] >= 15:
        drivers.append("High training load")
    if player_dict["nutrition"] <= 68:
        drivers.append("Suboptimal nutrition profile")

    return drivers[:4]

def likely_injury_zone_weighted(player_dict):

    fatigue_score = 0

    if player_dict["stress"] >= 70:
        fatigue_score += 1
    if player_dict["sleep_hours"] <= 6.5:
        fatigue_score += 1
    if player_dict["knee_strength"] <= 68:
        fatigue_score += 1
    if player_dict["hamstring_flexibility"] <= 68:
        fatigue_score += 1
    if player_dict["balance_score"] <= 75:
        fatigue_score += 1

    if fatigue_score >= 3:
        return "General Fatigue / Whole Body"


    if player_dict["hamstring_flexibility"] <= 68:
        return "Hamstring"

    if player_dict["knee_strength"] <= 68:
        return "Knee"

    if player_dict["balance_score"] <= 78:
        return "Ankle"

    if player_dict["training_hours"] >= 15:
        return "Quadriceps"

    if player_dict["previous_injuries"] >= 2:
        return "Previously Affected Lower Limb"

    return "Lower Limb"
