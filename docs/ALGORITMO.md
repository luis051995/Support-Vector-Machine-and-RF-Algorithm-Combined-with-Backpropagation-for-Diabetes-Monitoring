# Revisión del algoritmo

## Errores encontrados en el proyecto recibido

1. Entrena en orden Edad, Sexo, Peso, Altura, Antecedentes, Acetona. La inferencia original coloca ppm en la primera posición, seguido de edad y otras variables: no conserva el contrato.
2. La red original recibe características más score SVM y probabilidad RF durante el entrenamiento; la inferencia añade solamente score SVM, omitiendo RF.
3. El AUC original se calcula con etiquetas binarias y no con puntuaciones continuas.
4. Convierte alturas solo si el máximo de toda la columna es menor a 10; el CSV mezcla metros y centímetros y esa condición no corrige las filas individuales.
5. SMOTE y escalado se ejecutan antes de la validación interna de GridSearchCV. La validación puede ver información de las particiones a través del preprocesamiento.
6. La red combina salidas de modelos que ya vieron esas mismas filas, lo que favorece sobreajuste.

## Nueva implementación

Se aplica limpieza por fila, se excluyen observaciones fuera del dominio adulto y se eliminan duplicados exactos. No se inventa una corrección de peso o edad: se exportan los excluidos para revisión. El CSV original se conserva íntegro.

Orden explícito de variables en entrenamiento e inferencia. Estandarización dentro del pipeline SVM. Sin SMOTE; se utilizan pesos de clase en SVM/RF. Ajuste SVM de 6 combinaciones con validación de 4 grupos estratificados dentro del entrenamiento. La red MLP recibe dos columnas de predicciones fuera de muestra, una de SVM y otra de RF. Se entrena con backpropagation a través de MLPClassifier; sustituye la red TensorFlow y elimina esa dependencia.

Se aparta una quinta parte aproximada mediante StratifiedGroupKFold, semilla 42. La prueba no se utiliza para ajustar hiperparámetros ni el umbral (0.5 fijo). Para las predicciones fuera de muestra de la combinación se usan grupos disjuntos. El ajuste de hiperparámetros comparte el conjunto de desarrollo: no se interpreta como validación anidada completa del proceso; la estimación reportada es exclusivamente la del holdout reservado.

Sin identificadores reales, se agrupa de forma conservadora por edad, sexo, peso y altura para no separar mediciones con el mismo perfil demográfico. Esto no garantiza que una persona con cambios de peso no aparezca en dos grupos. La siguiente versión requiere IDs seudonimizados reales y validación externa. No se presupone que los registros sean de personas únicas.

## Resultados de esta ejecución

286 filas originales; 3 alturas convertidas; 4 filas excluidas; 6 duplicados retirados; 276 utilizables. Entrenamiento: 220; prueba: 56 (41 etiqueta 0 y 15 etiqueta 1).

| Modelo | Exactitud | AUC | F1 | Sensibilidad | Especificidad |
|---|---:|---:|---:|---:|---:|
| SVM de referencia | 85.71 % | 0.9301 | 0.7143 | 66.67 % | 92.68 % |
| RF de referencia | 91.07 % | 0.9301 | 0.8148 | 73.33 % | 97.56 % |
| SVM + RF + MLP | 92.86 % | 0.9512 | 0.8462 | 73.33 % | 100.00 % |

Matriz híbrida: TN=41, FP=0, FN=4, TP=11. Especificidad 100 % corresponde solo a 41 negativos del holdout y no garantiza ese resultado en población real. Son comparaciones de las referencias implementadas en esta versión; no se ejecutó una reproducción íntegra de la red original con TensorFlow. No afirmar una mejora porcentual frente al sistema original sin un protocolo común.

No se optimizó sensibilidad ni se calibraron probabilidades para uso clínico. La puntuación corresponde al comportamiento de las etiquetas recibidas; la calidad y procedencia de esas etiquetas no están verificadas. No usar la salida para decisiones de tratamiento ni para descartar enfermedad.

## Siguiente evaluación de investigación

Obtener consentimiento y procedencia de datos; identificador de paciente y sesión, diagnóstico de referencia verificado, fecha, protocolo y calibración de sensor; separar por paciente y centro; estimar intervalos de confianza y calibración; fijar el umbral clínico solo con datos de desarrollo; verificar de forma externa en una cohorte independiente. No sustituir la variable acetona por otra base de diabetes sin esa variable: sería otro modelo.

Fuentes técnicas: https://scikit-learn.org/stable/common_pitfalls.html y https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.StackingClassifier.html
