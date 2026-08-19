import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow.keras import layers, models

st.set_page_config(
    page_title='Clasificador de Imagenes | UTH',
    page_icon='*',
    layout='centered'
)

CLASES = ['Avion', 'Auto', 'Pajaro', 'Gato', 'Ciervo',
          'Perro', 'Rana', 'Caballo', 'Barco', 'Camion']

@st.cache_resource
def load_model():
    m = models.Sequential([
        layers.Conv2D(32, (3,3), activation='relu', padding='same', input_shape=(32,32,3)),
        layers.MaxPooling2D(2,2),
        layers.Dropout(0.25),
        layers.Conv2D(64, (3,3), activation='relu', padding='same'),
        layers.MaxPooling2D(2,2),
        layers.Dropout(0.25),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(10, activation='softmax')
    ])
    m.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
    m.load_weights('cifar10_weights.weights.h5')
    return m

model = load_model()

st.title('Clasificador de Imagenes - Henry Monge')
st.caption('Universidad Tecnologica de Honduras (UTH) | Computacion en la Nube')
st.divider()

st.subheader('Sube una imagen o toma una foto')
fuente = st.radio('Fuente:', ['Subir archivo', 'Camara'], horizontal=True)
img_input = None

if fuente == 'Subir archivo':
    uploaded = st.file_uploader('Selecciona imagen', type=['jpg', 'jpeg', 'png', 'webp'])
    if uploaded:
        img_input = Image.open(uploaded).convert('RGB')
else:
    foto = st.camera_input('Tomar foto con camara')
    if foto:
        img_input = Image.open(foto).convert('RGB')

if img_input:
    col1, col2 = st.columns([1, 1])
    with col1:
        st.image(img_input, caption='Imagen cargada', use_column_width=True)
    with col2:
        img_resized = img_input.resize((32, 32))
        img_array = np.array(img_resized).astype('float32') / 255.0
        img_array = np.expand_dims(img_array, 0)
        with st.spinner('Analizando imagen...'):
            preds = model.predict(img_array, verbose=0)[0]
        top_idx = np.argmax(preds)
        clase = CLASES[top_idx]
        confianza = preds[top_idx]
        st.success('Resultado:')
        st.metric(label='Objeto detectado', value=clase, delta=f'{confianza:.1%} confianza')
        st.write('Top 3 predicciones:')
        for i in np.argsort(preds)[::-1][:3]:
            st.progress(float(preds[i]), text=f'{CLASES[i]}: {preds[i]:.1%}')

    with st.expander('Sobre el modelo'):
        st.write("""
        - Dataset: CIFAR-10
        - Arquitectura: CNN con bloques convolucionales + Dropout
        - Entrenado en: Google Colab (GPU T4)
        - Clases: Avion, Auto, Pajaro, Gato, Ciervo, Perro, Rana, Caballo, Barco, Camion
        """)

st.divider()
st.caption('Desarrollado por: Henry Monge | UTH | Dataset: CIFAR-10 | Framework: TensorFlow/Keras')
