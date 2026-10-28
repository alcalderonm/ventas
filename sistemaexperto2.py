
import random
from collections import Counter

from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# ==================================================
# 1. GENERACION DE DATOS SIMULADOS
# ==================================================

random.seed(42)

def calcular_riesgo(asistencia, promedio, actividades, reprobadas):
    """
    Etiquetas didacticas para crear datos simulados.
    No son criterios oficiales de ninguna universidad.
    """
    puntaje = 0

    if asistencia < 60:
        puntaje += 3
    elif asistencia < 75:
        puntaje += 2
    elif asistencia < 85:
        puntaje += 1

    if promedio < 2.5:
        puntaje += 3
    elif promedio < 3.0:
        puntaje += 2
    elif promedio < 3.5:
        puntaje += 1

    if actividades < 50:
        puntaje += 2
    elif actividades < 70:
        puntaje += 1

    if reprobadas >= 3:
        puntaje += 2
    elif reprobadas >= 1:
        puntaje += 1

    if puntaje >= 6:
        return "Alto"
    elif puntaje >= 3:
        return "Moderado"
    else:
        return "Bajo"


X = []
y = []

for _ in range(1200):
    asistencia = random.uniform(35, 100)
    promedio = random.uniform(1.0, 5.0)
    actividades = random.uniform(20, 100)
    reprobadas = random.randint(0, 6)

    X.append([
        asistencia,
        promedio,
        actividades,
        reprobadas
    ])

    y.append(
        calcular_riesgo(
            asistencia,
            promedio,
            actividades,
            reprobadas
        )
    )

# ==================================================
# 2. DIVISION DE DATOS
# ==================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)

# ==================================================
# 3. CREACION Y ENTRENAMIENTO DE LA RED NEURONAL
# ==================================================

modelo = make_pipeline(
    StandardScaler(),
    MLPClassifier(
        hidden_layer_sizes=(16, 8),
        activation="relu",
        solver="adam",
        learning_rate_init=0.001,
        max_iter=3000,
        early_stopping=False,
        random_state=42
    )
)

modelo.fit(X_train, y_train)
print("Entrenamiento finalizado.")

# ==================================================
# 4. EVALUACION DEL MODELO
# ==================================================

predicciones = modelo.predict(X_test)

print("\n===== EVALUACION DEL MODELO =====")
print("Exactitud:", round(
    accuracy_score(y_test, predicciones), 3
))

print("\nInforme de clasificacion:")
print(classification_report(
    y_test,
    predicciones,
    zero_division=0
))

etiquetas = ["Alto", "Bajo", "Moderado"]

print("Matriz de confusion:")
print(confusion_matrix(
    y_test,
    predicciones,
    labels=etiquetas
))

# ==================================================
# 5. MOTOR DE INFERENCIA DEL SISTEMA EXPERTO
# ==================================================

def evaluar_reglas(asistencia, promedio, actividades, reprobadas):
    alertas = []
    recomendaciones = []

    if asistencia < 60:
        alertas.append("Asistencia inferior al 60 %.")
        recomendaciones.append(
            "Revisar las causas de inasistencia."
        )

    if promedio < 3.0:
        alertas.append("Promedio inferior a 3.0.")
        recomendaciones.append(
            "Proponer tutorias y seguimiento docente."
        )

    if actividades < 50:
        alertas.append("Baja entrega de actividades.")
        recomendaciones.append(
            "Revisar dificultades de acceso y cumplimiento."
        )

    if reprobadas >= 3:
        alertas.append("Tres o mas asignaturas reprobadas.")
        recomendaciones.append(
            "Realizar una revision integral del caso."
        )

    if not alertas:
        alertas.append(
            "No se activaron las reglas de alerta definidas."
        )
        recomendaciones.append(
            "Mantener el seguimiento academico habitual."
        )

    return alertas, recomendaciones

# ==================================================
# 6. INGRESO Y ANALISIS DE ESTUDIANTES
# ==================================================

def leer_numero(mensaje, minimo, maximo, entero=False):
    while True:
        try:
            valor = float(input(mensaje))

            if not minimo <= valor <= maximo:
                print(
                    f"Ingrese un valor entre {minimo} y {maximo}."
                )
                continue

            if entero and not valor.is_integer():
                print("Debe ingresar un numero entero.")
                continue

            return int(valor) if entero else valor

        except ValueError:
            print("Entrada invalida. Intente nuevamente.")


def analizar_estudiante():
    print("\n===== NUEVO ANALISIS =====")

    nombre = input("Nombre o codigo del estudiante: ").strip()

    if not nombre:
        print("El identificador no puede estar vacio.")
        return

    asistencia = leer_numero(
        "Asistencia (0-100 %): ", 0, 100
    )

    promedio = leer_numero(
        "Promedio academico (0-5): ", 0, 5
    )

    actividades = leer_numero(
        "Actividades entregadas (0-100 %): ", 0, 100
    )

    reprobadas = leer_numero(
        "Asignaturas reprobadas (0-10): ", 0, 10,
        entero=True
    )

    datos = [[
        asistencia,
        promedio,
        actividades,
        reprobadas
    ]]

    # Prediccion neuronal
    categoria = modelo.predict(datos)[0]
    probabilidades = modelo.predict_proba(datos)[0]
    clases = modelo.named_steps["mlpclassifier"].classes_

    # Inferencia basada en reglas
    alertas, recomendaciones = evaluar_reglas(
        asistencia,
        promedio,
        actividades,
        reprobadas
    )

    print("\n===== INFORME DEL SISTEMA EXPERTO =====")
    print("Estudiante:", nombre)
    print("Categoria estimada por la red:", categoria)

    print("\nProbabilidades estimadas:")
    for clase, probabilidad in zip(clases, probabilidades):
        print(f"{clase}: {probabilidad * 100:.1f}%")

    print("\nReglas y alertas:")
    for alerta in alertas:
        print("-", alerta)

    print("\nRecomendaciones:")
    for recomendacion in recomendaciones:
        print("-", recomendacion)

    print(
        "\nNOTA: resultado experimental para apoyo y revision docente."
    )


while True:
    print("\n===== MENU PRINCIPAL =====")
    print("1. Analizar estudiante")
    print("2. Consultar distribucion de datos de entrenamiento")
    print("3. Salir")

    opcion = input("Seleccione una opcion: ").strip()

    if opcion == "1":
        analizar_estudiante()

    elif opcion == "2":
        print("\nClases en los datos simulados:")
        for clase, cantidad in sorted(Counter(y).items()):
            print(f"{clase}: {cantidad} registros")

    elif opcion == "3":
        print("Programa finalizado.")
        break

    else:
        print("Opcion no valida.")
