
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Datos: [asistencia (%), nota promedio]
X = [
    [95, 4.5], [90, 4.0], [80, 3.8], [85, 4.2],
    [60, 2.8], [50, 2.2], [45, 1.8], [55, 2.5],
    [75, 3.5], [65, 3.0], [92, 4.4], [48, 2.0]
]

# 0 = sin alerta; 1 = requiere revisión
y = [0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 0, 1]

# Separar entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# Crear la red y escalar los datos
modelo = make_pipeline(
    StandardScaler(),
    MLPClassifier(
        hidden_layer_sizes=(8,),
        solver="adam",
        learning_rate_init=0.01,
        max_iter=2000,
        random_state=42
    )
)

# Entrenar
modelo.fit(X_train, y_train)

# Evaluar
predicciones = modelo.predict(X_test)
print("Exactitud:", accuracy_score(y_test, predicciones))

# Analizar un nuevo caso
estudiante = [[55, 2.4]]
print("Clasificación:", modelo.predict(estudiante)[0])
