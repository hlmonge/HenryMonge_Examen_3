import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# ── Configuración ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title='Clasificador de Imágenes | UTH',
    page_icon='🎯',
    layout='centered'
)

CLASES = ['Avión', 'Auto', 'Pájaro', 'Gato', 'Ciervo',
          'Perro', 'Rana', 'Caballo', 'Barco', 'Camión']

EMOJIS = {
    'Avión': '✈️', 'Auto': '🚗', 'Pájaro': '🐦', 'Gato': '🐱',
    'Ciervo': '🦌', 'Perro': '🐶', 'Rana': '🐸', 'Caballo': '🐴',
    'Barco': '🚢', 'Camión': '🚛'
}

# ── Cargar modelo (cache para no recargar en cada interacción) ─────────────────
@st.cache_resource
def load_model():
    return tf.keras.models.load_model('cifar10_model.keras')

model = load_model()

# ── Header ────────────────────────────────────────────────────────────────────
st.title('🎯 Clasificador de Imágenes - Henry Monge')
st.caption('Universidad Tecnológica de Honduras (UTH) | Computación en la Nube')
st.divider()

# ── Input de imagen ───────────────────────────────────────────────────────────
st.subheader('📤 Sube una imagen o toma una foto')
fuente = st.radio('Fuente:', ['📁 Subir archivo', '📷 Cámara'], horizontal=True)

img_input = None

if fuente == '📁 Subir archivo':
    uploaded = st.file_uploader('Selecciona imagen', type=['jpg', 'jpeg', 'png', 'webp'])
    if uploaded:
        img_input = Image.open(uploaded).convert('RGB')
else:
    foto = st.camera_input('Tomar foto con cámara')
    if foto:
        img_input = Image.open(foto).convert('RGB')

# ── Predicción ────────────────────────────────────────────────────────────────
if img_input:
    col1, col2 = st.columns([1, 1])

    with col1:
        st.image(img_input, caption='Imagen cargada', use_column_width=True)

    with col2:
        # Preprocesar igual que en entrenamiento
        img_resized = img_input.resize((32, 32))
        img_array = np.array(img_resized).astype('float32') / 255.0
        img_array = np.expand_dims(img_array, 0)  # (1, 32, 32, 3)

        with st.spinner('🔍 Analizando imagen...'):
            preds = model.predict(img_array, verbose=0)[0]
            top_idx = np.argmax(preds)
            clase = CLASES[top_idx]
            confianza = preds[top_idx]

        emoji = EMOJIS.get(clase, '🔍')
        st.success('**Resultado:**')
        st.metric(label=f'{emoji} Objeto detectado', value=clase, delta=f'{confianza:.1%} confianza')

        st.write('**Top 3 predicciones:**')
        top3 = np.argsort(preds)[::-1][:3]
        for i in top3:
            label = f'{EMOJIS.get(CLASES[i], "")} {CLASES[i]}: {preds[i]:.1%}'
            st.progress(float(preds[i]), text=label)

    # Info extra
    with st.expander('ℹ️ Sobre el modelo'):
        st.write("""
        - **Dataset:** CIFAR-10 (50,000 imágenes de entrenamiento)
        - **Arquitectura:** CNN con 3 bloques convolucionales + BatchNorm + Dropout
        - **Entrenado en:** Google Colab (GPU T4)
        - **Clases:** Avión, Auto, Pájaro, Gato, Ciervo, Perro, Rana, Caballo, Barco, Camión
        """)

st.divider()
st.caption('🎓 Desarrollado por: **[TU NOMBRE]** | UTH 2024 | Dataset: CIFAR-10 | Framework: TensorFlow/Keras')
