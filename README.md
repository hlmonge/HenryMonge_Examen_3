# 🎯 Clasificador de Imágenes — CIFAR-10

**Universidad Tecnológica de Honduras (UTH)**  
**Examen: Computación en la Nube | Docente: Ing. Asalia Zavala**  
**Desarrollado por:** Henry Luis Monge

---

## ¿Qué hace la app?

Aplicación web que identifica objetos en imágenes usando un modelo de Machine Learning entrenado con el dataset CIFAR-10. El usuario sube una foto o usa la cámara, y la app devuelve el objeto detectado con su nivel de confianza.

**Clases que reconoce:** Avión, Auto, Pájaro, Gato, Ciervo, Perro, Rana, Caballo, Barco, Camión.

---

## 🏗️ Arquitectura del modelo

- **Tipo:** CNN (Red Neuronal Convolucional)
- **Bloques:** 3 bloques Conv2D + BatchNormalization + MaxPooling + Dropout
- **Clasificador:** Dense(256) → Dense(10, softmax)
- **Optimizador:** Adam con ReduceLROnPlateau
- **Entrenamiento:** Google Colab (GPU T4), ~30 épocas con EarlyStopping
- **Data augmentation:** Flip horizontal, shift, rotación

---

## 🛠️ Herramientas utilizadas

| Herramienta | Uso |
|-------------|-----|
| Google Colab | Entrenamiento del modelo (GPU T4 gratis) |
| TensorFlow/Keras | Framework de ML |
| CIFAR-10 (keras.datasets) | Dataset de entrenamiento |
| Streamlit | Desarrollo de la app web |
| Streamlit Cloud | Despliegue gratuito |
| GitHub | Repositorio del código |

---

## 📂 Estructura del proyecto

```
├── CIFAR10_Clasificador.ipynb   # Notebook de entrenamiento (Google Colab)
├── app.py                       # Aplicación Streamlit
├── cifar10_model.keras          # Modelo entrenado (generado en Colab)
├── requirements.txt             # Dependencias
└── README.md                    # Esta documentación
```

---

## 🚀 Cómo usar la app

1. Ir a la URL pública de la app (Streamlit Cloud): https://henrymonge-examen3.streamlit.app/
2. Seleccionar fuente: **Subir archivo** o **Cámara**
3. Cargar o tomar una foto
4. La app muestra el objeto detectado y el porcentaje de confianza

---

## ⚙️ Cómo reproducir el entrenamiento

1. Abrir `CIFAR10_Clasificador.ipynb` en Google Colab
2. Ir a **Runtime → Change runtime type → GPU (T4)**
3. Ejecutar todas las celdas en orden
4. Al finalizar, descarga `cifar10_model.keras`
5. Subir el modelo al repositorio GitHub junto con `app.py`

---

## 🌐 Despliegue en Streamlit Cloud

1. Crear repo en GitHub con: `app.py`, `cifar10_model.keras`, `requirements.txt`, `README.md`
2. Ir a [share.streamlit.io](https://share.streamlit.io)
3. New app → conectar repo → Main file: `app.py`
4. Deploy → copiar URL pública y entregar

---

## 📊 Resultados del modelo

- **Dataset:** CIFAR-10 (50,000 train / 10,000 test)
- **Accuracy en test:** ~80–85% (varía por época/seed)
- **Tamaño modelo:** ~15–20 MB

---

*Nota: El modelo fue entrenado con imágenes de 32×32 px. Para mejores resultados, usa imágenes claras y bien enfocadas del objeto a clasificar.*
