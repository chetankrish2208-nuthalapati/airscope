import joblib
import pandas as pd
import streamlit as st
import folium

from PIL import Image
from streamlit_folium import st_folium


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
            caption="Uploaded image",
            width="stretch"
        )

        st.info(
            "The image is used as visual context. "
            "It does not directly measure AQI."
        )

    except Exception as error:
        st.error(f"Could not open the image: {error}")


if "prediction_result" not in st.session_state:
    st.session_state.prediction_result = None

if "confidence_result" not in st.session_state:
    st.session_state.confidence_result = None


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
        values = pd.DataFrame(
            [[input_values.get(feature, 0) for feature in features]],
            columns=features
        )

        prediction = model.predict(values)[0]
        probabilities = model.predict_proba(values)[0]
        confidence = float(max(probabilities))

        st.session_state.prediction_result = prediction
        st.session_state.confidence_result = confidence

    except Exception as error:
        st.session_state.prediction_result = None
        st.session_state.confidence_result = None
        st.error(f"Prediction failed: {error}")


prediction = st.session_state.prediction_result
confidence = st.session_state.confidence_result


if prediction is not None:

    prediction_text = str(prediction).strip().lower()

    if prediction_text == "high":
        st.error("🔴 High estimated pollution risk")
        st.write(
            "Consider reducing prolonged outdoor activity "
            "and checking official local AQI information."
        )

    elif prediction_text == "moderate":
        st.warning("🟡 Moderate estimated pollution risk")
        st.write(
            "Use caution during long outdoor activities "
            "and monitor official updates."
        )

    else:
        st.success("🟢 Low estimated pollution risk")
        st.write(
            "Current numerical indicators suggest lower risk, "
            "but continue monitoring conditions."
        )

    st.write(f"Model confidence: {confidence:.1%}")

    st.subheader("Location")

    map_object = folium.Map(
        location=[latitude, longitude],
        zoom_start=10
    )

    folium.Marker(
        [latitude, longitude],
        popup=f"Estimated risk: {prediction}",
        tooltip="Selected location"
    ).add_to(map_object)

    st_folium(
        map_object,
        width=700,
        height=450
    )


st.divider()

st.caption(
    "Educational prototype. Not an official AQI service, "
    "medical tool, or emergency-warning system."
)
