# 🎯 Clasificador de Imágenes — CIFAR-10

**Universidad Tecnológica de Honduras (UTH)**  
**Examen: Computación en la Nube | Docente: Ing. Asalia Zavala**  
**Desarrollado por:** Henry Luis Monge

**Link sitio Streamlit:** https://henrymonge-examen3.streamlit.app/

**Link repo:** https://github.com/hlmonge/HenryMonge_Examen_3
---

## ¿Qué hace la app?

Aplicación web que identifica objetos en imágenes usando un modelo de Machine Learning entrenado con el dataset CIFAR-10. El usuario sube una foto o usa la cámara, y la app devuelve el objeto detectado con su nivel de confianza.

**Clases que reconoce:** Avión, Auto, Pájaro, Gato, Ciervo, Perro, Rana, Caballo, Barco, Camión.

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
├── app.py                       # Aplicación Streamlit
├── cifar10_weights.weights.h5   # Modelo entrenado (generado en Colab)
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

## 📊 Resultados del modelo

- **Dataset:** CIFAR-10 (50,000 train / 10,000 test)
- **Accuracy en test:** ~80–85% (varía por época/seed)
- **Tamaño modelo:** ~15–20 MB

---

*Nota: El modelo fue entrenado con imágenes de 32×32 px. Para mejores resultados, usa imágenes claras y bien enfocadas del objeto a clasificar.*
