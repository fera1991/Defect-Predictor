# ==============================
# Configuración de Parámetros
# ==============================

# Parámetros Generales
GENERAL_PARAMS = {
    'RANDOM_STATE': 42,
    'DATA_FRACTION': 0.6,  # Recomendación: Aumentar a 0.5 o 1.0 tras validar
    'DATA_TYPE': 1,
}

DATASET_SIZES = {
    'TRAIN_VAL_PERCENT': 0.20,  # El dataset para validación
    'TRAIN_TEST_PERCENT': 0.20,  # El dataset para prueba
    'TEST_CLASS1_RATIO': 0.25, # Proporción de la clase 1 en el dataset de prueba
    'DATA_FRACTION': 1.0,  # El dataset para entrenamiento
}

# Parámetros para TfidfVectorizer
TFIDF_PARAMS = {
    'max_features': 12000,  # Aumentado para capturar más términos
    'min_df': 1,  # Reducido para incluir términos menos frecuentes
    'ngram_range': (1,3),  # Limitado a unigramas y bigramas para menos ruido
    'max_df': 0.7,  # Ligeramente más estricto
    'stop_words': None,  # Sin stop words para simplicidad
    'lowercase': True,  # Mantenido para consistencia
    'analyzer': 'word',
    'norm': 'l2',
    'sublinear_tf': True,
    'strip_accents': 'unicode',
    'token_pattern': r'\b\w+\b',  # Sin cambios
}

# Parámetros para TruncatedSVD
SVD_PARAMS = {
    'n_components': 50,  # Aumentado para más variabilidad
    'algorithm': 'randomized',
    'n_iter': 10,
    'random_state': GENERAL_PARAMS['RANDOM_STATE']
}

# Parámetros para RandomForestClassifier
RF_PARAMS = {
    'n_estimators': 500,
    'max_depth': None,
    'min_samples_split': 2,
    'min_samples_leaf': 1,
    'max_features': 'sqrt',  # Usar raíz cuadrada de características
    'class_weight': 'balanced_subsample',
    'random_state': GENERAL_PARAMS['RANDOM_STATE'],
    'n_jobs': 2,  # Usar múltiples núcleos para entrenamiento
    'max_samples': 0.9,  # Usar 90% de las muestras por árbol
}

# Parámetros para XGBClassifier (agregado para pipeline.py)
XGB_PARAMS = {
    'n_estimators': 500,  # Más árboles para mayor capacidad
    'max_depth': 15,  # Mayor profundidad para capturar patrones complejos
    'learning_rate': 0.05,  # Aprendizaje más lento para mejor convergencia
    'subsample': 0.8,  # Usar más datos por árbol
    'colsample_bytree': 0.7,  # Más características por árbol
    'reg_lambda': 2.0,  # Regularización moderada
    'eval_metric': 'logloss',
    'random_state': GENERAL_PARAMS['RANDOM_STATE'],
    'n_jobs': 2,  # Usar múltiples núcleos para entrenamiento
    'gamma': 0.3,  # Penalizar divisiones poco útiles
}

# Parámetros para SimpleImputer
IMPUTER_PARAMS = {
    'strategy': 'median'
}

# Parámetros para StandardScaler
SCALER_PARAMS = {
    'with_mean': True,
    'with_std': True
}

# Parámetros para OneHotEncoder
OHE_PARAMS = {
    'handle_unknown': 'infrequent_if_exist',  # Manejo robusto de categorías raras
    'min_frequency': 0.01,  # Considerar categorías con al menos 1% de frecuencia
}

# Rangos para búsqueda de hiperparámetros (para RandomizedSearchCV)
HYPERPARAM_RANGES_UPDATED = {
    'TFIDF': {
        'preprocessor__text_full__tfidf__max_features': [3000,6000, 12000],
        'preprocessor__text_full__tfidf__min_df': [1, 2, 3],
        'preprocessor__text_full__tfidf__max_df': [0.7, 0.8, 0.9],
        'preprocessor__text_full__tfidf__ngram_range': [(1, 1), (1, 2),(1,3)],
        'preprocessor__text_full__tfidf__sublinear_tf': [True, False],
    },
    'SVD': {
        'preprocessor__text_full__svd__n_components': [50, 75, 100],
        'preprocessor__text_full__svd__n_iter': [5, 10, 15],
    },
    'RandomForest': {
        'clf__estimator__n_estimators': [100, 300, 500],
        'clf__estimator__max_depth': [10, 20, None],
        'clf__estimator__min_samples_split': [2, 5, 10],
        'clf__estimator__min_samples_leaf': [1, 5, 10],
        'clf__estimator__max_features': ['sqrt', 'log2'],
    },
    'XGBoost': {
        'clf__estimator__n_estimators': [100, 300, 500],
        'clf__estimator__learning_rate': [0.01, 0.05, 0.1],
        'clf__estimator__max_depth': [6, 10, 15],
        'clf__estimator__subsample': [0.7, 0.8, 1.0],
        'clf__estimator__colsample_bytree': [0.7, 0.8, 1.0],
        'clf__estimator__reg_lambda': [1.0, 1.5, 2.0],
        'clf__estimator__gamma': [0.0, 0.1, 0.3],
    },
    'VotingClassifier': {
        'clf__voting': ['hard', 'soft'],
        'clf__weights': [(1,1), (2,1), (1,2)],
    },
}