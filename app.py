import joblib
import pandas as pd
import streamlit as st
import folium

from PIL import Image
from streamlit_folium import folium_static


st.set_page_config(
    page_title="AirScope",
    page_icon="🌍",
    layout="wide"
)


st.title("🌍 AirScope")
st.subheader("Multimodal air-quality risk prototype")

st.write(
    "AirScope combines pollutant measurements with environmental image "
    "context to estimate regional pollution risk."
)


@st.cache_resource
def load_model():
    return joblib.load("air_quality_model.joblib")


try:
    bundle = load_model()

    model = bundle["model"]
    features = list(bundle["features"])

except Exception as error:
    st.error(f"Could not load the air-quality model: {error}")
    st.stop()


st.sidebar.header("Environmental data")


pm25 = st.sidebar.number_input(
    "PM2.5",
    min_value=0.0,
    value=40.0
)

pm10 = st.sidebar.number_input(
    "PM10",
    min_value=0.0,
    value=80.0
)

no2 = st.sidebar.number_input(
    "NO2",
    min_value=0.0,
    value=30.0
)

so2 = st.sidebar.number_input(
    "SO2",
    min_value=0.0,
    value=20.0
)

co = st.sidebar.number_input(
    "CO",
    min_value=0.0,
    value=2.0
)

o3 = st.sidebar.number_input(
    "O3",
    min_value=0.0,
    value=60.0
)

temperature = st.sidebar.number_input(
    "Temperature (°C)",
    value=28.0
)

humidity = st.sidebar.number_input(
    "Humidity (%)",
    min_value=0.0,
    max_value=100.0,
    value=60.0
)

wind_speed = st.sidebar.number_input(
    "Wind Speed",
    min_value=0.0,
    value=8.0
)

latitude = st.sidebar.number_input(
    "Latitude",
    value=17.3850,
    format="%.4f"
)

longitude = st.sidebar.number_input(
    "Longitude",
    value=78.4867,
    format="%.4f"
)


uploaded_image = st.file_uploader(
    "Upload an environmental image",
    type=["jpg", "jpeg", "png"]
)


if uploaded_image is not None:

    try:
        image = Image.open(uploaded_image)

        st.image(
            image,
            caption="Uploaded environmental image",
            width="stretch"
        )

    except Exception as error:
        st.error(f"Could not open the image: {error}")


if "prediction_result" not in st.session_state:
    st.session_state.prediction_result = None

if "confidence_result" not in st.session_state:
    st.session_state.confidence_result = None

if "probability_table" not in st.session_state:
    st.session_state.probability_table = None


if st.button("Estimate pollution risk", type="primary"):

    input_values = {
        "pm25": pm25,
        "pm10": pm10,
        "no2": no2,
        "so2": so2,
        "co": co,
        "o3": o3,
        "temperature": temperature,
        "humidity": humidity,
        "wind_speed": wind_speed
    }

    try:
        input_data = pd.DataFrame(
            [[input_values.get(feature, 0) for feature in features]],
            columns=features
        )

        prediction = model.predict(input_data)[0]

        if hasattr(model, "predict_proba"):

            probabilities = model.predict_proba(input_data)[0]
            class_names = model.classes_

            probability_table = pd.DataFrame({
                "Risk Level": class_names,
                "Probability": probabilities
            })

            probability_table["Probability"] = (
                probability_table["Probability"] * 100
            ).round(2)

            probability_table["Probability"] = (
                probability_table["Probability"].astype(str) + "%"
            )

            confidence = float(max(probabilities))

        else:
            probability_table = pd.DataFrame({
                "Risk Level": [prediction],
                "Probability": ["Unavailable"]
            })

            confidence = None

        st.session_state.prediction_result = prediction
        st.session_state.confidence_result = confidence
        st.session_state.probability_table = probability_table

    except Exception as error:
        st.session_state.prediction_result = None
        st.session_state.confidence_result = None
        st.session_state.probability_table = None

        st.error(f"Prediction failed: {error}")


prediction = st.session_state.prediction_result
confidence = st.session_state.confidence_result
probability_table = st.session_state.probability_table


if prediction is not None:

    prediction_text = str(prediction).strip().lower()

    if prediction_text == "high":
        st.error("🔴 High estimated pollution risk")

    elif prediction_text == "moderate":
        st.warning("🟡 Moderate estimated pollution risk")

    elif prediction_text == "low":
        st.success("🟢 Low estimated pollution risk")

    else:
        st.info(f"Estimated pollution risk: {prediction}")

    if confidence is not None:
        st.write(f"Model confidence: {confidence:.2%}")

    st.subheader("📊 Probability by risk level")

    if probability_table is not None:
        st.dataframe(
            probability_table,
            hide_index=True,
            width="stretch"
        )

    st.subheader("📍 Selected location")

    st.write(
        f"Latitude: {latitude:.4f} | Longitude: {longitude:.4f}"
    )

    map_object = folium.Map(
        location=[latitude, longitude],
        zoom_start=10
    )

    folium.Marker(
        location=[latitude, longitude],
        popup=f"Pollution risk: {prediction}",
        tooltip="Selected location",
        icon=folium.Icon(color="red", icon="cloud")
    ).add_to(map_object)

    folium_static(
        map_object,
        width=700,
        height=500
    )


st.divider()

st.caption(
    "Educational prototype. Not an official AQI service, "
    "medical tool, or emergency-warning system."
)
