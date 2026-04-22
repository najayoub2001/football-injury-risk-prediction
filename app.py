import streamlit as st
import pandas as pd
import joblib
import os
import uuid
from streamlit_cropper import st_cropper
from PIL import Image, ImageDraw, ImageOps
from utils.helpers import (
    build_model_input,
    risk_band,
    risk_colour,
    zone_marker,
    simulate_30_day_trend,
    scale_1_to_9,
    descale_to_1_to_9,
    readiness_score,
    get_risk_drivers,
    likely_injury_zone_weighted
)

st.set_page_config(
    page_title="Football Injury Monitoring System",
    page_icon="⚽",
    layout="wide"
)

st.markdown("""
<style>
html, body, [class*="css"] {
    background-color: #0b1220;
    color: #e5e7eb;
}
section[data-testid="stSidebar"] {
    background-color: #111827;
}
h1, h2, h3 {
    color: #f3f4f6;
}
.card {
    background: linear-gradient(180deg, #111827 0%, #0f172a 100%);
    padding: 1rem;
    border-radius: 16px;
    border: 1px solid #1f2937;
    box-shadow: 0 6px 20px rgba(0,0,0,0.25);
    margin-bottom: 1rem;
}
.small-muted {
    color: #9ca3af;
    font-size: 0.9rem;
}
</style>
""", unsafe_allow_html=True)


SQUAD_CSV = "squad_players.csv"
ASSETS_DIR = "assets"
CUSTOM_IMG_DIR = os.path.join(ASSETS_DIR, "custom_players")
DEFAULT_AVATAR = os.path.join(ASSETS_DIR, "avatar_default.png")
BODY_IMG = os.path.join(ASSETS_DIR, "body_front.png")

os.makedirs(CUSTOM_IMG_DIR, exist_ok=True)




def normalize_player_name(name: str) -> str:
    return " ".join(name.strip().split())

def save_uploaded_player_image(uploaded_file, player_name: str, cropped_img=None) -> str:
    if uploaded_file is None and cropped_img is None:
        return DEFAULT_AVATAR

    ext = ".png"
    safe_name = "".join(
        c for c in normalize_player_name(player_name).lower().replace(" ", "_")
        if c.isalnum() or c == "_"
    )
    unique_name = f"{safe_name}_{uuid.uuid4().hex[:8]}{ext}"
    save_path = os.path.join(CUSTOM_IMG_DIR, unique_name)

    if cropped_img is not None:
        final_img = cropped_img.convert("RGB")
    else:
        final_img = Image.open(uploaded_file).convert("RGB")


    final_img = ImageOps.fit(final_img, (247, 247), method=Image.Resampling.LANCZOS)
    final_img.save(save_path, format="PNG")

    return save_path.replace("\\", "/")


def delete_player_image_if_custom(image_path: str):
    if not image_path:
        return

    normalized_path = image_path.replace("\\", "/")
    custom_dir_normalized = CUSTOM_IMG_DIR.replace("\\", "/")
    default_avatar_normalized = DEFAULT_AVATAR.replace("\\", "/")


    if (
        normalized_path != default_avatar_normalized
        and normalized_path.startswith(custom_dir_normalized)
        and os.path.exists(normalized_path)
    ):
        try:
            os.remove(normalized_path)
        except Exception as e:
            st.warning(f"Player deleted, but image cleanup failed: {e}")


def draw_body_hotspot(image_path: str, zone_name: str):
    if zone_name == "No Major Vulnerability Detected":
        green_path = "assets/body_front_green.png"
        if os.path.exists(green_path):
            return Image.open(green_path).convert("RGBA")
        return Image.open(image_path).convert("RGBA")

    if zone_name == "General Fatigue / Whole Body":
        red_path = "assets/body_front_red.png"
        if os.path.exists(red_path):
            return Image.open(red_path).convert("RGBA")
        return Image.open(image_path).convert("RGBA")

    img = Image.open(image_path).convert("RGBA")
    draw = ImageDraw.Draw(img)

    x, y = zone_marker(zone_name)

    glow_r = 26
    core_r = 8

    draw.ellipse(
        (x - glow_r, y - glow_r, x + glow_r, y + glow_r),
        fill=(239, 68, 68, 70)
    )
    draw.ellipse(
        (x - 14, y - 14, x + 14, y + 14),
        outline=(255, 255, 255, 220),
        width=3
    )
    draw.ellipse(
        (x - core_r, y - core_r, x + core_r, y + core_r),
        fill=(239, 68, 68, 255)
    )

    return img

def ensure_required_columns(df: pd.DataFrame) -> pd.DataFrame:
    expected = [
        "player_name", "position", "age", "height_cm", "weight_kg",
        "training_hours", "matches_played", "previous_injuries",
        "knee_strength", "hamstring_flexibility", "reaction_time",
        "balance_score", "sprint_speed", "agility", "sleep_hours",
        "stress", "nutrition", "warmup", "minutes_played",
        "games_played", "days_since_last_injury", "sprint_distance_ratio",
        "competition_density_7", "competition_density_28", "avatar"
    ]
    for col in expected:
        if col not in df.columns:
            df[col] = None
    return df

def load_squad() -> pd.DataFrame:
    df = pd.read_csv(SQUAD_CSV)
    df = ensure_required_columns(df)
    return df

def save_squad(df: pd.DataFrame):
    df.to_csv(SQUAD_CSV, index=False)

def update_player_in_csv(updated_player: dict):
    df = load_squad()

    target_name = normalize_player_name(updated_player["player_name"])

    mask = df["player_name"].fillna("").apply(normalize_player_name) == target_name

    if mask.any():
        for key, value in updated_player.items():
            if key in df.columns:
                df.loc[mask, key] = value

        save_squad(df)
        return True

    return False

def compute_squad_risk_df(squad_df: pd.DataFrame) -> pd.DataFrame:
    cards = []
    for _, row in squad_df.iterrows():
        player = row.to_dict()
        X = build_model_input(player)
        prob = float(model.predict_proba(X)[0][1])
        cards.append({
            "Player": row["player_name"],
            "Position": row["position"],
            "Age": int(row["age"]),
            "Risk Probability": prob,
            "Risk Level": risk_band(prob),
            "Club Games": int(row["games_played"]),
            "Minutes": int(row["minutes_played"])
        })
    return pd.DataFrame(cards).sort_values("Risk Probability", ascending=False)


model = joblib.load("injury_prediction_model.pkl")
squad = load_squad()


page = st.sidebar.selectbox(
    "Select Page",
    ["Squad Overview", "Player Dashboard", "Add New Player", "Compare Players", "Manage Players", "Model Insights"]
)

selected_name = st.sidebar.selectbox("Select Player", squad["player_name"].tolist())
player_row = squad[squad["player_name"] == selected_name].iloc[0].to_dict()

if st.session_state.get("selected_player_last") != selected_name:
    st.session_state["selected_player_last"] = selected_name
    st.session_state["injury_zone"] = likely_injury_zone_weighted(player_row)


if page == "Squad Overview":
    st.title("Squad Injury Monitoring Dashboard")

    risk_df = compute_squad_risk_df(squad)

    filter_options = ["All", "Low", "Moderate", "High"]
    selected_filter = st.selectbox("Filter by Risk Band", filter_options)

    if selected_filter != "All":
        filtered_risk_df = risk_df[risk_df["Risk Level"] == selected_filter].copy()
    else:
        filtered_risk_df = risk_df.copy()

    top1, top2, top3 = st.columns(3)
    top1.metric("Players Monitored", len(risk_df))
    top2.metric("Average Risk", f"{risk_df['Risk Probability'].mean():.2f}")
    top3.metric("High-Risk Players", int((risk_df["Risk Level"] == "High").sum()))

    st.subheader("Top 5 Highest-Risk Players")
    top5 = risk_df.head(5)[["Player", "Risk Probability", "Risk Level"]]
    st.dataframe(top5, use_container_width=True)

    st.subheader("Squad Risk Ranking")
    st.dataframe(filtered_risk_df, use_container_width=True)

    st.subheader("Risk Distribution")
    chart_df = filtered_risk_df.set_index("Player")["Risk Probability"]
    st.bar_chart(chart_df)


elif page == "Player Dashboard":
    st.title("Player Dashboard")
    header_placeholder = st.container()


    selected_player_key = normalize_player_name(player_row["player_name"])


    required_keys = [
        f"age_{selected_player_key}",
        f"weight_{selected_player_key}",
        f"minutes_{selected_player_key}",
        f"games_{selected_player_key}"
    ]


    if (st.session_state.get("editor_player_last") != selected_player_key or
            any(k not in st.session_state for k in required_keys)):
        st.session_state["editor_player_last"] = selected_player_key
        st.session_state[f"age_{selected_player_key}"] = int(player_row["age"])
        st.session_state[f"weight_{selected_player_key}"] = float(player_row["weight_kg"])
        st.session_state[f"minutes_{selected_player_key}"] = int(player_row["minutes_played"])
        st.session_state[f"games_{selected_player_key}"] = int(player_row["games_played"])


    editable_player = player_row.copy()
    editable_player["age"] = int(st.session_state[f"age_{selected_player_key}"])
    editable_player["weight_kg"] = float(st.session_state[f"weight_{selected_player_key}"])
    editable_player["minutes_played"] = int(st.session_state[f"minutes_{selected_player_key}"])
    editable_player["games_played"] = int(st.session_state[f"games_{selected_player_key}"])
    editable_player["matches_played"] = int(st.session_state[f"games_{selected_player_key}"])  # keep synced


    dashboard_left, dashboard_right = st.columns([1.1, 2.2])

    with dashboard_left:
        st.subheader("Player Inputs")

        editable_player["training_hours"] = st.slider(
            "Training Load (hours/week)",
            1, 30, int(player_row["training_hours"]),
            key=f"training_{selected_player_key}",
            help="Estimated weekly training volume. Higher values can increase cumulative fatigue and injury risk."
        )

        editable_player["previous_injuries"] = st.slider(
            "Previous Injuries",
            0, 8, int(player_row["previous_injuries"]),
            key=f"previnj_{selected_player_key}",
            help="Number of prior injuries recorded for the player. Repeated injuries often increase future risk."
        )

        editable_player["sprint_speed"] = st.slider(
            "10m Sprint Time (seconds)",
            5.0, 7.5, float(player_row["sprint_speed"]),
            key=f"sprint_{selected_player_key}",
            help="Time taken to cover 10 metres. Lower values indicate better acceleration."
        )

        editable_player["competition_density_7"] = st.slider(
            "Matches Played (Last 7 Days)",
            0, 7, int(player_row["competition_density_7"]),
            key=f"comp7_{selected_player_key}",
            help="Short-term match congestion indicator. More matches in the past week can increase acute fatigue."
        )

        editable_player["competition_density_28"] = st.slider(
            "Matches Played (Last 28 Days)",
            0, 20, int(player_row["competition_density_28"]),
            key=f"comp28_{selected_player_key}",
            help="Longer-term fixture density indicator across the last four weeks."
        )

        sleep_ui = st.slider(
            "Sleep Quality (1–9)", 1, 9,
            descale_to_1_to_9(player_row["sleep_hours"], 5.0, 10.0),
            key=f"sleep_{selected_player_key}",
            help="Higher values indicate better sleep and recovery quality."
        )
        stress_ui = st.slider(
            "Stress Level (1–9)", 1, 9,
            descale_to_1_to_9(player_row["stress"], 20, 90),
            key=f"stress_{selected_player_key}",
            help="Higher values indicate greater physical or psychological stress load."
        )
        knee_ui = st.slider(
            "Knee Strength (1–9)", 1, 9,
            descale_to_1_to_9(player_row["knee_strength"], 60, 90),
            key=f"knee_{selected_player_key}",
            help="Higher values indicate better lower-limb strength and joint stability."
        )
        hamstring_ui = st.slider(
            "Hamstring Flexibility (1–9)", 1, 9,
            descale_to_1_to_9(player_row["hamstring_flexibility"], 60, 90),
            key=f"ham_{selected_player_key}",
            help="Higher values indicate better hamstring mobility and lower posterior-chain vulnerability."
        )
        balance_ui = st.slider(
            "Balance / Control (1–9)", 1, 9,
            descale_to_1_to_9(player_row["balance_score"], 70, 95),
            key=f"bal_{selected_player_key}",
            help="Higher values indicate stronger balance, coordination, and neuromuscular control."
        )
        agility_ui = st.slider(
            "Agility / Explosiveness (1–9)", 1, 9,
            descale_to_1_to_9(player_row["agility"], 60, 90),
            key=f"agility_{selected_player_key}",
            help="Higher values indicate better movement sharpness and change-of-direction ability."
        )
        nutrition_ui = st.slider(
            "Nutrition Quality (1–9)", 1, 9,
            descale_to_1_to_9(player_row["nutrition"], 60, 90),
            key=f"nutri_{selected_player_key}",
            help="Higher values indicate better nutritional support for performance and recovery."
        )
        warmup_ui = st.slider(
            "Warm-up Compliance (1–9)", 1, 9,
            descale_to_1_to_9(player_row["warmup"], 0, 10),
            key=f"warmup_{selected_player_key}",
            help="Higher values indicate better adherence to warm-up and injury-prevention routines."
        )

        editable_player["sleep_hours"] = scale_1_to_9(sleep_ui, 5.0, 10.0)
        editable_player["stress"] = scale_1_to_9(stress_ui, 20, 90)
        editable_player["knee_strength"] = scale_1_to_9(knee_ui, 60, 90)
        editable_player["hamstring_flexibility"] = scale_1_to_9(hamstring_ui, 60, 90)
        editable_player["balance_score"] = scale_1_to_9(balance_ui, 70, 95)
        editable_player["agility"] = scale_1_to_9(agility_ui, 60, 90)
        editable_player["nutrition"] = scale_1_to_9(nutrition_ui, 60, 90)
        editable_player["warmup"] = scale_1_to_9(warmup_ui, 0, 10)

        editable_player["reaction_time"] = st.slider(
            "Reaction Time (ms)",
            150, 400, int(player_row["reaction_time"]),
            key=f"react_{selected_player_key}",
            help="Reaction speed measured in milliseconds. Lower values indicate faster response time."
        )


    current_prob = float(model.predict_proba(build_model_input(editable_player))[0][1])
    current_band = risk_band(current_prob)

    if current_band == "Low":
        current_zone = "No Major Vulnerability Detected"
        current_img = draw_body_hotspot(BODY_IMG, current_zone)
    else:
        current_zone = likely_injury_zone_weighted(editable_player)
        current_img = draw_body_hotspot(BODY_IMG, current_zone)

    trend_df = simulate_30_day_trend(editable_player, model)


    with header_placeholder:

        head1, head2, head3, head4, head5, head6 = st.columns([1.5, 1, 1, 1, 1, 1])

        with head1:
            avatar_path = editable_player.get("avatar", DEFAULT_AVATAR)
            if not avatar_path or not os.path.exists(avatar_path):
                avatar_path = DEFAULT_AVATAR

            avatar_col, info_col = st.columns([1, 2])

            with avatar_col:
                st.image(avatar_path, width=90)

            with info_col:
                st.markdown(f"""
                <div class="card">
                    <h3 style="margin-bottom:0.3rem;">{editable_player['player_name']}</h3>
                    <div class="small-muted">{editable_player['position']}</div>
                </div>
                """, unsafe_allow_html=True)

        with head2:
            st.metric("Weight", f"{int(editable_player['weight_kg'])} kg")

        with head3:
            st.metric("Age", int(editable_player["age"]))

        with head4:
            st.metric("Minutes Played", int(editable_player["minutes_played"]))

        with head5:
            st.metric("Games Played", int(editable_player["games_played"]))

        with head6:
            st.metric("Risk", f"{current_prob:.2f}")

    with dashboard_right:
        st.markdown("### Injury Risk Monitor")
        c1, c2 = st.columns([2, 1])

        with c1:
            chart = trend_df.set_index("Day")[["Injury Risk", "Training Hours"]]
            st.line_chart(chart)

        with c2:
            ready_score = readiness_score(editable_player)

            st.markdown(f"""
            <div class="card">
                <h3>Risk Summary</h3>
                <p><strong>Probability:</strong> {current_prob:.2f}</p>
                <p><strong>Risk Band:</strong> {current_band}</p>
                <p><strong>Readiness Score:</strong> {ready_score}/100</p>
                <p><strong>Likely Vulnerability:</strong> {current_zone}</p>
            </div>
            """, unsafe_allow_html=True)

            if current_band == "Low":
                st.success("✓ Player in Stable Condition")
                st.markdown("**System Status:** `No major biomechanical vulnerability detected`")
                caption_text = "Player Condition: Optimal"

            elif current_zone == "General Fatigue / Whole Body":
                st.warning("⚠ Monitor Player Load")
                st.markdown("**System Alert:** `Whole-body fatigue profile detected`")
                caption_text = "Whole-Body Fatigue Risk Detected"

            else:
                st.markdown(
                    f"**Biomechanical Vulnerability Detected:** `{current_zone}`"
                )

                if current_band == "High":
                    st.error("⚠ Elevated Injury Risk Detected")
                else:
                    st.warning("⚠ Monitor Player Load")

                caption_text = f"Likely Vulnerability: {current_zone}"

            st.image(current_img, caption=caption_text, width=280)

        b1, b2 = st.columns(2)

        with b1:
            st.subheader("Key Risk Drivers")
            indicators = get_risk_drivers(editable_player)

            if indicators:
                for i in indicators:
                    st.write(f"- {i}")
            else:
                st.write("- No major risk drivers detected.")

        with b2:
            st.subheader("Recommended Action")
            if current_band == "Low":
                st.success("Maintain current training and continue standard monitoring.")
            elif current_band == "Moderate":
                st.warning("Review workload and recovery indicators. Consider reducing load slightly.")
            else:
                st.error("High alert. Reduce workload, review recovery, and investigate lower-limb vulnerability.")


    st.markdown("### Player Profile Editor")
    st.caption("Edit profile values here, then save them to the squad file.")

    edit_col1, edit_col2 = st.columns(2)

    with edit_col1:
        st.number_input(
            "Age",
            min_value=16,
            max_value=45,
            key=f"age_{selected_player_key}",
            help="Player age in years."
        )

        st.number_input(
            "Weight (kg)",
            min_value=45.0,
            max_value=130.0,
            step=0.1,
            key=f"weight_{selected_player_key}",
            help="Player body weight in kilograms."
        )

    with edit_col2:
        st.number_input(
            "Minutes Played",
            min_value=0,
            max_value=5000,
            key=f"minutes_{selected_player_key}",
            help="Total minutes played by the player."
        )

        st.number_input(
            "Games Played (season)",
            min_value=0,
            max_value=60,
            key=f"games_{selected_player_key}",
            help="Total games played by the player this season. This also updates the model's season games input."
        )

    if st.button("Save Player Changes"):
        save_player = player_row.copy()

        save_player["age"] = int(st.session_state[f"age_{selected_player_key}"])
        save_player["weight_kg"] = float(st.session_state[f"weight_{selected_player_key}"])
        save_player["minutes_played"] = int(st.session_state[f"minutes_{selected_player_key}"])
        save_player["games_played"] = int(st.session_state[f"games_{selected_player_key}"])
        save_player["matches_played"] = int(st.session_state[f"games_{selected_player_key}"])

        fields_to_update = [
            "age",
            "weight_kg",
            "minutes_played",
            "games_played",
            "matches_played",
            "training_hours",
            "previous_injuries",
            "sprint_speed",
            "competition_density_7",
            "competition_density_28",
            "sleep_hours",
            "stress",
            "knee_strength",
            "hamstring_flexibility",
            "balance_score",
            "agility",
            "nutrition",
            "warmup",
            "reaction_time"
        ]

        for field in fields_to_update:
            if field in editable_player:
                save_player[field] = editable_player[field]

        updated = update_player_in_csv(save_player)

        if updated:
            st.success(f"Changes saved for {save_player['player_name']}.")
            st.rerun()
        else:
            st.error("Could not update the player record.")

elif page == "Add New Player":
    st.title("Add New Player")
    st.write("Create a new player profile and add them directly to the squad.")


    if "new_player_uploaded_image_name" not in st.session_state:
        st.session_state["new_player_uploaded_image_name"] = None

    if "new_player_cropped_image" not in st.session_state:
        st.session_state["new_player_cropped_image"] = None

    st.subheader("Player Photo")

    uploaded_image = st.file_uploader(
        "Upload Player Photo (crop to 247 × 247 avatar)",
        type=["png", "jpg", "jpeg"],
        key="new_player_photo_uploader"
    )

    if uploaded_image is not None:
        if st.session_state["new_player_uploaded_image_name"] != uploaded_image.name:
            st.session_state["new_player_uploaded_image_name"] = uploaded_image.name
            st.session_state["new_player_cropped_image"] = None

        raw_image = Image.open(uploaded_image).convert("RGB")

        st.caption("Center the player image inside the square crop area.")
        cropped_player_image = st_cropper(
            raw_image,
            aspect_ratio=(1, 1),
            box_color="#22c55e",
            realtime_update=True,
            key="new_player_cropper"
        )

        if cropped_player_image is not None:
            preview_img = ImageOps.fit(
                cropped_player_image,
                (247, 247),
                method=Image.Resampling.LANCZOS
            )
            st.session_state["new_player_cropped_image"] = preview_img

        if st.session_state["new_player_cropped_image"] is not None:
            st.caption("Final avatar preview (247 × 247)")
            st.image(st.session_state["new_player_cropped_image"], width=140)

    st.divider()

    with st.form("add_player_form", clear_on_submit=False):
        col1, col2 = st.columns(2)

        with col1:
            new_name = st.text_input("Player Name")
            new_position = st.selectbox("Position", ["Goalkeeper", "Defender", "Midfielder", "Forward"])
            new_age = st.number_input("Age", min_value=16, max_value=45, value=21)
            new_height = st.number_input("Height (cm)", min_value=150, max_value=220, value=177)
            new_weight = st.number_input("Weight (kg)", min_value=45, max_value=130, value=73)
            new_minutes = st.number_input("Minutes Played", min_value=0, max_value=5000, value=0)
            new_games = st.number_input("Games Played", min_value=0, max_value=60, value=0)

        with col2:
            new_training = st.slider("Training Load (hours/week)", 1, 30, 10)
            new_matches_played = st.slider("Games Played (season)", 0, 60, 22)
            new_previous_injuries = st.slider("Previous Injuries", 0, 8, 1)
            new_sprint = st.slider("10m Sprint Time (seconds)", 5.0, 7.5, 5.9)
            new_comp7 = st.slider("Matches Played (Last 7 Days)", 0, 7, 1)
            new_comp28 = st.slider("Matches Played (Last 28 Days)", 0, 20, 4)
            new_reaction = st.slider("Reaction Time (ms)", 150, 400, 249)

        st.subheader("Readiness and Physical Inputs")
        col3, col4 = st.columns(2)

        with col3:
            new_sleep_ui = st.slider("Sleep Quality (1–9)", 1, 9, 5)
            new_stress_ui = st.slider("Stress Level (1–9)", 1, 9, 5)
            new_knee_ui = st.slider("Knee Strength (1–9)", 1, 9, 5)
            new_hamstring_ui = st.slider("Hamstring Flexibility (1–9)", 1, 9, 5)

        with col4:
            new_balance_ui = st.slider("Balance / Control (1–9)", 1, 9, 5)
            new_agility_ui = st.slider("Agility / Explosiveness (1–9)", 1, 9, 5)
            new_nutrition_ui = st.slider("Nutrition Quality (1–9)", 1, 9, 5)
            new_warmup_ui = st.slider("Warm-up Compliance (1–9)", 1, 9, 5)

        add_player_btn = st.form_submit_button("Add Player to Squad")

    def build_new_player_dict(clean_name: str, save_avatar_path: str):
        return {
            "player_name": clean_name,
            "position": new_position,
            "age": int(new_age),
            "height_cm": float(new_height),
            "weight_kg": float(new_weight),
            "training_hours": int(new_training),
            "matches_played": int(new_matches_played),
            "previous_injuries": int(new_previous_injuries),
            "knee_strength": scale_1_to_9(new_knee_ui, 60, 90),
            "hamstring_flexibility": scale_1_to_9(new_hamstring_ui, 60, 90),
            "reaction_time": int(new_reaction),
            "balance_score": scale_1_to_9(new_balance_ui, 70, 95),
            "sprint_speed": float(new_sprint),
            "agility": scale_1_to_9(new_agility_ui, 60, 90),
            "sleep_hours": scale_1_to_9(new_sleep_ui, 5.0, 10.0),
            "stress": scale_1_to_9(new_stress_ui, 20, 90),
            "nutrition": scale_1_to_9(new_nutrition_ui, 60, 90),
            "warmup": scale_1_to_9(new_warmup_ui, 0, 10),
            "minutes_played": int(new_minutes),
            "games_played": int(new_games),
            "days_since_last_injury": 30,
            "sprint_distance_ratio": 1.0,
            "competition_density_7": int(new_comp7),
            "competition_density_28": int(new_comp28),
            "avatar": save_avatar_path
        }

    if add_player_btn:
        clean_name = normalize_player_name(new_name)
        existing_names = squad["player_name"].fillna("").apply(normalize_player_name).tolist()

        if not clean_name:
            st.error("Please enter a player name.")
        elif clean_name in existing_names:
            st.error("A player with this name already exists in the squad.")
        else:
            avatar_path = save_uploaded_player_image(
                uploaded_image,
                clean_name,
                st.session_state.get("new_player_cropped_image")
            )

            new_player = build_new_player_dict(clean_name, avatar_path)

            save_df = load_squad()
            save_df = pd.concat([save_df, pd.DataFrame([new_player])], ignore_index=True)
            save_squad(save_df)

            preview_prob = float(model.predict_proba(build_model_input(new_player))[0][1])
            preview_band = risk_band(preview_prob)
            preview_ready = readiness_score(new_player)

            st.success(f"{clean_name} has been added to the squad.")

            st.session_state["new_player_uploaded_image_name"] = None
            st.session_state["new_player_cropped_image"] = None

            st.markdown(f"""
            <div class="card">
                <h3>New Player Preview</h3>
                <p><strong>Name:</strong> {clean_name}</p>
                <p><strong>Predicted Risk:</strong> {preview_prob:.2f} ({preview_band})</p>
                <p><strong>Readiness Score:</strong> {preview_ready}/100</p>
            </div>
            """, unsafe_allow_html=True)

            st.rerun()


elif page == "Compare Players":
    st.title("Compare Players")

    player_names = squad["player_name"].tolist()

    col_left, col_right = st.columns(2)
    with col_left:
        p1_name = st.selectbox("Select Player 1", player_names, key="compare_p1")
    with col_right:
        p2_name = st.selectbox("Select Player 2", player_names, key="compare_p2")

    p1 = squad[squad["player_name"] == p1_name].iloc[0].to_dict()
    p2 = squad[squad["player_name"] == p2_name].iloc[0].to_dict()

    p1_prob = float(model.predict_proba(build_model_input(p1))[0][1])
    p2_prob = float(model.predict_proba(build_model_input(p2))[0][1])

    p1_ready = readiness_score(p1)
    p2_ready = readiness_score(p2)

    c1, c2 = st.columns(2)

    with c1:
        avatar1 = p1.get("avatar", DEFAULT_AVATAR)
        if not avatar1 or not os.path.exists(avatar1):
            avatar1 = DEFAULT_AVATAR
        st.image(avatar1, width=100)
        st.markdown(f"""
        <div class="card">
            <h3>{p1_name}</h3>
            <p><strong>Position:</strong> {p1['position']}</p>
            <p><strong>Risk:</strong> {p1_prob:.2f} ({risk_band(p1_prob)})</p>
            <p><strong>Readiness:</strong> {p1_ready}/100</p>
            <p><strong>Minutes:</strong> {int(p1['minutes_played'])}</p>
            <p><strong>Games:</strong> {int(p1['games_played'])}</p>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        avatar2 = p2.get("avatar", DEFAULT_AVATAR)
        if not avatar2 or not os.path.exists(avatar2):
            avatar2 = DEFAULT_AVATAR
        st.image(avatar2, width=100)
        st.markdown(f"""
        <div class="card">
            <h3>{p2_name}</h3>
            <p><strong>Position:</strong> {p2['position']}</p>
            <p><strong>Risk:</strong> {p2_prob:.2f} ({risk_band(p2_prob)})</p>
            <p><strong>Readiness:</strong> {p2_ready}/100</p>
            <p><strong>Minutes:</strong> {int(p2['minutes_played'])}</p>
            <p><strong>Games:</strong> {int(p2['games_played'])}</p>
        </div>
        """, unsafe_allow_html=True)


elif page == "Manage Players":
    st.title("Manage Players")

    st.subheader("All Players in Squad")
    st.dataframe(
        squad[["player_name", "position", "age", "minutes_played", "games_played"]],
        use_container_width=True
    )

    delete_name = st.selectbox("Select Player to Delete", squad["player_name"].tolist())

    if "pending_delete_player" not in st.session_state:
        st.session_state["pending_delete_player"] = None

    if st.button("Delete Player from Squad"):
        st.session_state["pending_delete_player"] = delete_name

    if st.session_state["pending_delete_player"] is not None:
        st.warning(
            f"Are you sure you want to permanently delete "
            f"{st.session_state['pending_delete_player']}?"
        )
        st.caption("This will also remove the custom player image if one exists.")

        confirm_col, cancel_col = st.columns(2)

        with confirm_col:
            if st.button("Confirm Delete"):
                delete_clean = normalize_player_name(st.session_state["pending_delete_player"])

                player_to_delete = squad[
                    squad["player_name"].fillna("").apply(normalize_player_name) == delete_clean
                ]

                if not player_to_delete.empty:
                    avatar_path = player_to_delete.iloc[0].get("avatar", "")
                    delete_player_image_if_custom(avatar_path)

                updated_squad = squad[
                    squad["player_name"].fillna("").apply(normalize_player_name) != delete_clean
                ].copy()

                save_squad(updated_squad.drop(
                    columns=[col for col in ["source"] if col in updated_squad.columns],
                    errors="ignore"
                ))

                st.session_state["pending_delete_player"] = None
                st.success(f"{delete_name} has been removed from the squad.")
                st.rerun()

        with cancel_col:
            if st.button("Cancel Delete"):
                st.session_state["pending_delete_player"] = None
                st.info("Deletion cancelled.")

elif page == "Model Insights":
    st.title("Model Insights")

    results_df = pd.DataFrame({
        "Model": ["Random Forest", "LSTM"],
        "Accuracy": [0.95, 0.7875],
        "Precision": [0.95, 0.7949],
        "Recall": [0.95, 0.7750],
        "F1 Score": [0.95, 0.7848],
        "ROC-AUC": [0.9935, 0.8602]
    })

    st.subheader("Performance Comparison")
    st.dataframe(results_df, use_container_width=True)

    st.subheader("F1 Score Comparison")
    st.bar_chart(results_df.set_index("Model")["F1 Score"])

    st.subheader("Random Forest Feature Importance")
    feature_importance = pd.DataFrame({
        "Feature": [
            "Stress_Level_Score",
            "Balance_Test_Score",
            "Sleep_Hours_Per_Night",
            "Reaction_Time_ms",
            "Sprint_Speed_10m_s",
            "Hamstring_Flexibility",
            "Knee_Strength_Score",
            "Nutrition_Quality_Score",
            "Agility_Score",
            "Training_Hours_Per_Week"
        ],
        "Importance": [0.115, 0.108, 0.096, 0.090, 0.084, 0.080, 0.076, 0.072, 0.067, 0.061]
    })
    st.bar_chart(feature_importance.set_index("Feature"))

    st.subheader("Interpretation")
    st.write("""
- Random Forest outperformed LSTM across all metrics.
- The dataset is structured and tabular, which favours tree-based learning.
- The LSTM model was limited by the absence of true longitudinal player sequences.
- Stress, balance, sleep, reaction time, and sprint-related variables were the most influential predictors.
""")

    st.subheader("ROC Curve")
    try:
        st.image("roc_curve_comparison.png", caption="ROC Curve Comparison")
    except:
        st.info("Place roc_curve_comparison.png in the project folder to display it here.")
