# Defect-Predictor

**Defect-Predictor** es un proyecto enfocado en la predicción de defectos en software utilizando técnicas de aprendizaje automático. Este repositorio implementa modelos de machine learning para identificar módulos de software propensos a defectos, utilizando el dataset **Defectors** como base para el entrenamiento.

## Tabla de Contenidos
- [Requisitos](#requisitos)
- [Configuración del Entorno](#configuración-del-entorno)
- [Instalación de Dependencias](#instalación-de-dependencias)
- [Estructura del Proyecto](#estructura-del-proyecto)
  - [Descripción de Archivos](#descripción-de-archivos)
- [Uso](#uso)
- [Autores](#autores)
- [Reconocimientos](#reconocimientos)

## Requisitos
Para ejecutar este proyecto, necesitarás tener instalado lo siguiente:
- Python 3.8 o superior
- pip (gestor de paquetes de Python)
- Un entorno virtual (recomendado, como `venv` o `virtualenv`)
- Acceso al dataset **Defectors** (disponible en [Zenodo](https://zenodo.org/records/7708984))

## Configuración del Entorno
Sigue estos pasos para configurar el entorno de desarrollo:

1. **Clona el repositorio**:
   ```bash
   git clone https://github.com/fera1991/Defect-Predictor.git
   cd Defect-Predictor
   ```

2. **Crea un entorno virtual**:
   ```bash
   python -m venv venv
   ```

3. **Activa el entorno virtual**:
   - En Windows:
     ```bash
     .\venv\Scripts\activate
     ```
   - En macOS/Linux:
     ```bash
     source venv/bin/activate
     ```

4. **Verifica que el entorno está activo**:
   Deberías ver `(venv)` en tu terminal.

## Instalación de Biblioteas requeridas y comandos de instalación
Una vez activado el entorno virtual, instala las dependencias necesarias ejecutando los siguientes comandos:

- `numpy`: Para operaciones numéricas.
```bash
pip install numpy
```

- `pandas`: Para manipulación y análisis de datos.
```bash
pip install pandas
```

- `matplotlib`: Para visualización de datos.
```bash
pip install matplotlib
```

- `joblib`: Para facilitar la serialización (guardado y carga) de objetos de Python.
```bash
pip install joblib
```

- `scikit-learn`: Para implementar modelos de machine learning.
```bash
pip install scikit-learn
```

- `xgboost`: Para implementar algoritmos de aprendizaje automático.
```bash
pip install xgboost
```

- `imbalanced-learn`: Para trabajar con conjuntos de datos desbalanceados, es decir, cuando las clases en un problema de clasificación tienen una distribución desigual.
```bash
pip install imbalanced-learn
```

Para instalar todas las bibliotecas externas requeridas de una sola vez, ejecuta:
```bash
pip install numpy pandas matplotlib joblib scikit-learn xgboost imbalanced-learn
```

## Estructura del Proyecto
A continuación, se detalla la estructura del repositorio y la función de cada archivo o directorio principal:

### Descripción de Archivos
- **`jit_bug_prediction_splits/`**: Directorio que contiene los archivos descomprimidos del dataset **Defectors** que debe descargarse desde [Zenodo](https://zenodo.org/records/7708984) 
- **`line_bug_prediction_splits`**: Directorio que contiene los archivos de linea descomprimidos del dataset **Defectors** que debe descargarse desde [Zenodo](https://zenodo.org/records/7708984)
- **`evaluate_models.py`**: Scripts que contiene funciones para evaluar modelos de machine learning entrenados para predecir defectos en código.
- **`config.py`**: Script que define parámetros de configuración para el proyecto.
- **`metrics.py`**: Scripts con funciones para calcular métricas textuales y de complejidad del código fuente, como entropía de tokens, repetición de líneas, palabras por línea, palabras clave de error, puntos de decisión, niveles de indentación, y más.
- **`features.py`**: Define la función `extraer_features` para extraer características de un DataFrame, incluyendo métricas de código completo y diferencias (`content_full`, `content_diff`), características temporales (hora y día del commit), longitud de rutas de archivo, frecuencia de cambios por archivo y dominios de repositorios.
- **`pipeline.py`**: Script que crea un preprocesador (`ColumnTransformer`) que maneja características numéricas, categóricas y de texto.
- **`main.py`**: Script principal que orquesta el proceso de entrenamiento y evaluación. 
- **`utils.py`**: Script que contiene la función `encontrar_umbral_optimo_mcc` para calcular el umbral óptimo basado en el coeficiente de correlación de Matthews (MCC), evaluando diferentes umbrales para maximizar el equilibrio entre precisión y recall en predicciones binarias.
- **`train.py`**: Script para entrenar los modelos de machine learning. Incluye la implementación de algoritmos como Random Forest, SVM y redes neuronales.
- **`README.md`**: Este archivo, que proporciona una descripción general del proyecto y las instrucciones para su uso.

## Uso
1. **Descarga el dataset**:
   - Descarga el dataset **Defectors** desde [Zenodo](https://zenodo.org/records/7708984) y colócalo en el directorio `data/`.
   - Sigue las instrucciones del dataset para descomprimirlo y organizarlo correctamente.

2. **Ejecuta el proyecto desde archivo main**:
   ```bash
   python main.py
   ```
   > El script main maneja todo el proceso desde entrenamiento hasta evaluación e imprime cada paso que se realiza de cada archivo.

## Autores
El proyecto fue desarrollado por los siguientes autores:
- **Jonathan Ariel Cabrera Galdámez**.
- **Fernando Jose Galdámez Mendoza**.
- **Kevin Bryan Hernández López**.
- **Andrés Josué Mendoza Alvarado**.

## Reconocimientos
Este proyecto utiliza el dataset **Defectors**, disponible públicamente en [Zenodo](https://zenodo.org/records/7708984). Agradecemos a los autores del dataset por proporcionar un recurso valioso para la investigación en predicción de defectos de software. Este conjunto de datos contiene métricas de software y etiquetas de defectos que han sido fundamentales para el entrenamiento y evaluación de los modelos de este proyecto.