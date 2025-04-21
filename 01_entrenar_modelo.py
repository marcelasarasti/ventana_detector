import os
import cv2
import numpy as np
from skimage.feature import hog
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib

# === CONFIGURACIÓN ===
ventana_dir = "dataset/ventana"
no_ventana_dir = "dataset/no_ventana"
img_size = (128, 128)  # redimensionar imágenes

def load_images_from_folder(folder, label):
    features = []
    labels = []
    for filename in os.listdir(folder):
        path = os.path.join(folder, filename)
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            img = cv2.resize(img, img_size)
            hog_feat = hog(img,
                           orientations=9,
                           pixels_per_cell=(8, 8),
                           cells_per_block=(2, 2),
                           block_norm='L2-Hys',
                           transform_sqrt=True)
            features.append(hog_feat)
            labels.append(label)
    return features, labels

# Cargar datos
X_ventana, y_ventana = load_images_from_folder(ventana_dir, 1)
X_no_ventana, y_no_ventana = load_images_from_folder(no_ventana_dir, 0)

X = np.array(X_ventana + X_no_ventana)
y = np.array(y_ventana + y_no_ventana)

# Dividir dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Entrenar SVM
model = LinearSVC(max_iter=10000)
model.fit(X_train, y_train)

# Evaluar
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Exactitud: {acc:.2f}")
print(classification_report(y_test, y_pred))

# Guardar modelo entrenado
model_path = "modelo_ventana.pkl"
joblib.dump(model, model_path)
print(f"Modelo guardado en: {model_path}")
