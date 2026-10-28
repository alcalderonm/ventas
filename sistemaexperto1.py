
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# Datos simulados para entrenar la red neuronal
# Columnas: asistencia (%) y nota promedio
X = [
    [95, 4.5], [90, 4.0], [85, 3.8],
    [80, 3.5], [75, 3.2], [70, 3.0],
    [65, 2.8], [60, 2.5], [55, 2.2],
    [50, 2.0], [45, 1.8], [40, 1.5],
    [92, 4.3], [58, 2.7], [88, 3.9],
    [48, 2.1]
]

# 0 = sin alerta; 1 = requiere revisión
y = [
    0, 0, 0, 0, 0, 0,
    1, 1, 1, 1, 1, 1,
    0, 1, 0, 1
]

# Crear la red neuronal
modelo = make_pipeline(
    StandardScaler(),
    MLPClassifier(
        hidden_layer_sizes=(8,),
        max_iter=3000,
        random_state=42
    )
)

# Entrenar la red neuronal
modelo.fit(X, y)

print("SISTEMA DE APOYO AL RENDIMIENTO ACADEMICO")
print("------------------------------------------")

# Ingreso de datos
nombre = input("Nombre del estudiante: ").strip()

try:
    asistencia = float(input("Asistencia (0 a 100 %): "))
    nota = float(input("Nota promedio (0 a 5): "))

    # Validar los datos
    if not nombre:
        print("Error: debe ingresar el nombre.")
    elif not 0 <= asistencia <= 100:
        print("Error: la asistencia debe estar entre 0 y 100.")
    elif not 0 <= nota <= 5:
        print("Error: la nota debe estar entre 0 y 5.")
    else:
        # La red neuronal genera la clasificación
        datos = [[asistencia, nota]]
        resultado = modelo.predict(datos)[0]

        print("\nRESULTADO DEL ANALISIS")
        print("Estudiante:", nombre)
        print("Asistencia:", asistencia, "%")
        print("Nota promedio:", nota)

        if resultado == 1:
            print("Clasificacion: requiere revision academica.")
            print("Recomendacion: revisar el caso y ofrecer apoyo.")
        else:
            print("Clasificacion: sin alerta segun el modelo.")
            print("Recomendacion: continuar el seguimiento habitual.")

except ValueError:
    print("Error: ingrese valores numericos validos.")
