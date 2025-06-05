# ==============================
# Configuración de Parámetros
# ==============================

# Parámetros Generales
GENERAL_PARAMS = {
    'RANDOM_STATE': 42,
    'DATA_FRACTION': 0.3,  # Recomendación: Aumentar a 0.5 o 1.0 tras validar
    'DATA_TYPE': 1,
}

# Parámetros para TfidfVectorizer
TFIDF_PARAMS = {
    'max_features': 10000,
    'min_df': 1,
    'ngram_range': (1, 3),
    'max_df': 0.85,
    #'stop_words': ['def', 'import', 'class', 'if', 'for', 'while', 'return', 'try', 'except'],
    'lowercase': True,
    'analyzer': 'word',
    'norm': 'l2',
    'sublinear_tf': True
}

# Parámetros para TruncatedSVD
SVD_PARAMS = {
    'n_components': 100,
    'algorithm': 'randomized',
    'n_iter': 10,
    'random_state': GENERAL_PARAMS['RANDOM_STATE']
}

# Parámetros para RandomForestClassifier
RF_PARAMS = {
    'n_estimators': 200,
    'max_depth': 20,
    'min_samples_split': 5,
    'min_samples_leaf': 2,
    'max_features': 'sqrt',
    'class_weight': 'balanced',
    'random_state': GENERAL_PARAMS['RANDOM_STATE'],
    'n_jobs': -1
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
    'handle_unknown': 'ignore'
}