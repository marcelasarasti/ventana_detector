# 🪟 Detector de Ventanas basado en Geometría

Este proyecto detecta automáticamente una **ventana negra** en una pared de ladrillo a partir de un video. Utiliza únicamente **técnicas geométricas y morfológicas**, sin redes neuronales ni modelos preentrenados.

## 📹 ¿Cómo funciona?

1. Detecta ladrillos por su color naranja (espacio HSV)
2. Usa el tamaño real del ladrillo (33 × 23 cm) como **referencia de escala**
3. Detecta la ventana negra por umbral de color y morfología
4. Calcula:
   - Ancho en metros
   - Alto en metros
   - Área en m²
5. Muestra todo sobre el video en tiempo real

## 🔧 Requisitos

```bash
pip install -r requirements.txt
