# Detector de Ventanas usando Geometría y Homografía

Este proyecto implementa un sistema que detecta automáticamente una **ventana negra** en una pared de ladrillos, usando **técnicas geométricas y morfológicas**, sin redes neuronales. El sistema también **rectifica la vista de la ventana** usando homografía para facilitar la medición precisa.

---

## ¿Qué hace?

- Detecta **ladrillos naranjas** en una pared para usar como referencia de escala
- Detecta **ventanas negras** aplicando máscaras de color y operaciones morfológicas
- Calcula dimensiones reales de la ventana en **metros**
- Aplica una **homografía** para obtener una vista frontal de la ventana
- Muestra dos ventanas:
  - `Detección en video` con cuadros y medidas
  - `Ventana rectificada (Homografía)` con la ventana frontal

---
##  Requisitos

```bash
pip install -r requirements.txt
