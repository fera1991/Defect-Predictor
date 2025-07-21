import pandas as pd
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import joblib
import os
import logging
from sklearn.model_selection import RandomizedSearchCV
from sklearn.utils import resample
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, roc_auc_score, precision_recall_curve
from sklearn.inspection import permutation_importance
from sklearn.utils.validation import check_is_fitted
from features import extraer_features
from utils import encontrar_umbral_optimo_mcc
from config import GENERAL_PARAMS, DATASET_SIZES, HYPERPARAM_RANGES_UPDATED

# Configurar logging
logging.basicConfig(
    filename='train.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def perform_hyperparameter_search(pipeline, param_dist, X_train, y_train, search_type='wide'):
    logging.info(f"Realizando búsqueda {search_type}...")
    print(f"\nRealizando búsqueda {search_type}...")
    start_search = time.time()
    
    try:
        search = RandomizedSearchCV(
            pipeline,
            param_distributions=param_dist,
            n_iter=8,  # Reducido para evitar desbordamiento
            cv=2,       # Reducido para menor carga computacional
            scoring='f1_macro',
            random_state=GENERAL_PARAMS['RANDOM_STATE'],
            n_jobs=1,   # Usar 1 hilo
            verbose=10,  # Detalles por iteración
            error_score='raise',
        )
        search.fit(X_train, y_train)
        end_search = time.time()
        logging.info(f"Tiempo de búsqueda {search_type}: {end_search - start_search:.2f} segundos")
        logging.info(f"Mejores parámetros ({search_type}): {search.best_params_}")
        logging.info(f"Mejor F1-score ({search_type}): {search.best_score_:.4f}")
        print(f"🕒 Tiempo de búsqueda {search_type}: {end_search - start_search:.2f} segundos")
        print(f"Mejores parámetros ({search_type}): {search.best_params_}")
        print(f"Mejor F1-score ({search_type}): {search.best_score_:.4f}")
        return search.best_estimator_, search.best_params_, search.best_score_
    
    except OverflowError as e:
        logging.error(f"Error de desbordamiento en búsqueda {search_type}: {e}")
        print(f"❌ Error de desbordamiento en búsqueda {search_type}: {e}")
        raise
    except MemoryError as e:
        logging.error(f"Error de memoria en búsqueda {search_type}: {e}")
        print(f"❌ Error de memoria en búsqueda {search_type}: {e}")
        raise
    except Exception as e:
        logging.error(f"Error inesperado en búsqueda {search_type}: {e}")
        print(f"❌ Error inesperado en búsqueda {search_type}: {e}")
        raise

def load_data():
    logging.info("Cargando los datasets...")
    print("Cargando los datasets...")

    #   Rutas de los archivos
    data_paths = [
        "line_bug_prediction_splits/random/train.parquet.gzip",
        "line_bug_prediction_splits/random/test.parquet.gzip",
        "line_bug_prediction_splits/random/val.parquet.gzip",
        "jit_bug_prediction_splits/random/train.parquet.gzip",
        "jit_bug_prediction_splits/random/test.parquet.gzip",
        "jit_bug_prediction_splits/random/val.parquet.gzip"
    ]

    missing_files = []
    for path in data_paths:
        if not os.path.exists(path):
            missing_files.append(path)

    if missing_files:
        error_msg = f"Error: No se encontraron los siguientes archivos o carpetas:\n{', '.join(missing_files)}\nPor favor, asegúrese de que los archivos parquet estén en las rutas especificadas."
        logging.error(error_msg)
        print(error_msg)
        exit(1)        

    try:
        df_train_line = pd.read_parquet("line_bug_prediction_splits/random/train.parquet.gzip")
        df_test_line = pd.read_parquet("line_bug_prediction_splits/random/test.parquet.gzip")
        df_val_line = pd.read_parquet("line_bug_prediction_splits/random/val.parquet.gzip")
        df_train_jit = pd.read_parquet("jit_bug_prediction_splits/random/train.parquet.gzip")
        df_test_jit = pd.read_parquet("jit_bug_prediction_splits/random/test.parquet.gzip")
        df_val_jit = pd.read_parquet("jit_bug_prediction_splits/random/val.parquet.gzip")
        logging.info(f"Columnas en df_train_line: {df_train_line.columns.tolist()}")
        logging.info(f"Columnas en df_train_jit: {df_train_jit.columns.tolist()}")
        print("Columnas en df_train_line:", df_train_line.columns.tolist())
        print("Columnas en df_train_jit:", df_train_jit.columns.tolist())
        logging.info("Datasets cargados.")
        print("Datasets cargados.")
        return (df_train_line, df_test_line, df_val_line), (df_train_jit, df_test_jit, df_val_jit)
    except Exception as e:
        error_msg = f"Error al cargar los datasets: {e}\nPor favor, verifique que los archivos parquet existan y sean accesibles."
        logging.error(error_msg)
        print(error_msg)
        exit(1)    

def process_data(dfs_line, dfs_jit, data_type):
    df_train_line, df_test_line, df_val_line = dfs_line
    df_train_jit, df_test_jit, df_val_jit = dfs_jit
    
    logging.info("Fusionando datasets JIT y Line...")
    print("\nFusionando datasets JIT y Line...")
    df_train = df_train_jit.merge(df_train_line[['commit', 'repo', 'filepath', 'content']], 
                                 on=['commit', 'repo', 'filepath'], 
                                 suffixes=('_diff', '_full'))
    df_test = df_test_jit.merge(df_test_line[['commit', 'repo', 'filepath', 'content']], 
                               on=['commit', 'repo', 'filepath'], 
                               suffixes=('_diff', '_full'))
    df_val = df_val_jit.merge(df_val_line[['commit', 'repo', 'filepath', 'content']], 
                             on=['commit', 'repo', 'filepath'], 
                             suffixes=('_diff', '_full'))
    
    if data_type == 0:
        logging.info("Tipo 0: Usar datasets originales sin modificar ni estratificar")
        print("\nTipo 0: Usar datasets originales sin modificar ni estratificar")
        df_train['label'] = df_train['lines'].apply(lambda x: 0 if len(x) == 0 else 1)
        df_test['label'] = df_test['lines'].apply(lambda x: 0 if len(x) == 0 else 1)
        df_val['label'] = df_val['lines'].apply(lambda x: 0 if len(x) == 0 else 1)
    elif data_type == 1:
        logging.info("Tipo 1: Balancear train y val, mantener test desbalanceado")
        print("\nTipo 1: Balancear train y val, mantener test desbalanceado con tamaños basados en porcentajes")
        
        TRAIN_VAL_PERCENT = DATASET_SIZES['TRAIN_VAL_PERCENT']
        TRAIN_TEST_PERCENT = DATASET_SIZES['TRAIN_TEST_PERCENT']
        TEST_CLASS1_RATIO = DATASET_SIZES['TEST_CLASS1_RATIO']
        DATA_FRACTION = DATASET_SIZES['DATA_FRACTION']
        
        df_train['label'] = df_train['lines'].apply(lambda x: 0 if len(x) == 0 else 1)
        df_val['label'] = df_val['lines'].apply(lambda x: 0 if len(x) == 0 else 1)
        df_test['label'] = df_test['lines'].apply(lambda x: 0 if len(x) == 0 else 1)
        
        logging.info("Distribución de clases original:")
        logging.info(f"Entrenamiento ({len(df_train)} ejemplos): {df_train['label'].value_counts(normalize=True).to_dict()}")
        logging.info(f"Validación ({len(df_val)} ejemplos): {df_val['label'].value_counts(normalize=True).to_dict()}")
        logging.info(f"Prueba ({len(df_test)} ejemplos): {df_test['label'].value_counts(normalize=True).to_dict()}")
        print("\nDistribución de clases original:")
        print(f"Entrenamiento ({len(df_train)} ejemplos):", df_train['label'].value_counts(normalize=True))
        print(f"Validación ({len(df_val)} ejemplos):", df_val['label'].value_counts(normalize=True))
        print(f"Prueba ({len(df_test)} ejemplos):", df_test['label'].value_counts(normalize=True))
        
        total_train_size = len(df_train)
        target_val_size = int(total_train_size * TRAIN_VAL_PERCENT)
        target_test_size = int(total_train_size * TRAIN_TEST_PERCENT)
        
        logging.info("Ajustando conjunto de prueba...")
        print("\nAjustando conjunto de prueba...")
        test_0 = df_test[df_test['label'] == 0]
        test_1 = df_test[df_test['label'] == 1]
        target_class1_size = int(target_test_size * TEST_CLASS1_RATIO)
        target_class0_size = target_test_size - target_class1_size
        
        test_1_resampled = resample(
            test_1,
            n_samples=target_class1_size,
            replace=len(test_1) < target_class1_size,
            random_state=GENERAL_PARAMS['RANDOM_STATE']
        )
        
        test_0_resampled = resample(
            test_0,
            n_samples=target_class0_size,
            replace=len(test_0) < target_class0_size,
            random_state=GENERAL_PARAMS['RANDOM_STATE']
        )
        
        df_test = pd.concat([test_0_resampled, test_1_resampled])
        
        logging.info("Ajustando conjunto de validación...")
        print("\nAjustando conjunto de validación...")
        val_0 = df_val[df_val['label'] == 0]
        val_1 = df_val[df_val['label'] == 1]
        target_val_class_size = target_val_size // 2
        
        val_1_resampled = resample(
            val_1,
            n_samples=target_val_class_size,
            replace=len(val_1) < target_val_class_size,
            random_state=GENERAL_PARAMS['RANDOM_STATE']
        )
        
        val_0_resampled = resample(
            val_0,
            n_samples=target_val_class_size,
            replace=len(val_0) < target_val_class_size,
            random_state=GENERAL_PARAMS['RANDOM_STATE']
        )
        
        df_val_balanced = pd.concat([val_0_resampled, val_1_resampled])
        df_val = df_val_balanced.sample(frac=DATA_FRACTION, random_state=GENERAL_PARAMS['RANDOM_STATE'])
        
        logging.info("Balanceando entrenamiento...")
        print("\nBalanceando entrenamiento...")
        train_0 = df_train[df_train['label'] == 0]
        train_1 = df_train[df_train['label'] == 1]
        train_0_downsampled = resample(
            train_0, 
            n_samples=len(train_1), 
            random_state=GENERAL_PARAMS['RANDOM_STATE']
        )
        df_train_balanced = pd.concat([train_0_downsampled, train_1])
        df_train = df_train_balanced.sample(frac=DATA_FRACTION, random_state=GENERAL_PARAMS['RANDOM_STATE'])
        
        if abs(len(df_test) - target_test_size) > 10 or abs(df_test['label'].mean() - TEST_CLASS1_RATIO) > 0.01:
            logging.error(f"Conjunto de prueba no tiene ~{target_test_size} ejemplos o proporción incorrecta")
            raise ValueError(f"El conjunto de prueba no tiene ~{target_test_size} ejemplos o proporción incorrecta (esperado ~{TEST_CLASS1_RATIO*100}% clase 1)")
        if abs(df_val['label'].mean() - 0.5) > 0.01:
            logging.error("Conjunto de validación no está balanceado")
            raise ValueError("El conjunto de validación no está balanceado (~50%/50%)")
        
        logging.info("Distribución de clases final:")
        logging.info(f"Entrenamiento ({len(df_train)} ejemplos): {df_train['label'].value_counts(normalize=True).to_dict()}")
        logging.info(f"Validación ({len(df_val)} ejemplos): {df_val['label'].value_counts(normalize=True).to_dict()}")
        logging.info(f"Prueba ({len(df_test)} ejemplos): {df_test['label'].value_counts(normalize=True).to_dict()}")
        print("\nDistribución de clases final:")
        print(f"Entrenamiento ({len(df_train)} ejemplos):", df_train['label'].value_counts(normalize=True))
        print(f"Validación ({len(df_val)} ejemplos):", df_val['label'].value_counts(normalize=True))
        print(f"Prueba ({len(df_test)} ejemplos):", df_test['label'].value_counts(normalize=True))
    else:
        logging.error("data_type debe ser 0 o 1")
        raise ValueError("data_type debe ser 0 o 1")

    for df in [df_train, df_test, df_val]:
        df['content_diff'] = df['content_diff'].apply(
            lambda x: x.decode('utf-8', errors='ignore') if isinstance(x, bytes) else str(x)
        ).str.replace('\\n', '\n')
        df['content_full'] = df['content_full'].apply(
            lambda x: x.decode('utf-8', errors='ignore') if isinstance(x, bytes) else str(x)
        ).str.replace('\\n', '\n')

    X_train = df_train.drop(columns=['label'])
    y_train = df_train['label']
    X_test = df_test.drop(columns=['label'])
    y_test = df_test['label']
    X_val = df_val.drop(columns=['label'])
    y_val = df_val['label']

    logging.info(f"Tamaño del conjunto de entrenamiento: {len(df_train)}")
    logging.info(f"Tamaño del conjunto de prueba: {len(df_test)}")
    logging.info(f"Tamaño del conjunto de validación: {len(df_val)}")
    print(f"Tamaño del conjunto de entrenamiento: {len(df_train)}")
    print(f"Tamaño del conjunto de prueba: {len(df_test)}")
    print(f"Tamaño del conjunto de validación: {len(df_val)}")
    logging.info("Conjuntos de datos preparados.")
    print("Conjuntos de datos preparados.")
    
    return X_train, y_train, X_test, y_test, X_val, y_val, df_train, df_test, df_val

def extract_features(X_train, X_test, X_val, numeric_features, categorical_features):
    logging.info("Extrayendo características para el conjunto de entrenamiento...")
    print("\nExtrayendo características para el conjunto de entrenamiento...")
    start_extract = time.time()
    X_train_feats = extraer_features(X_train)
    if X_train_feats is None:
        logging.error("No se encontraron características para el conjunto de entrenamiento.")
        print("Error: No se encontraron características para el conjunto de entrenamiento.")
        exit(1)
    end_extract = time.time()
    logging.info(f"Tiempo de extracción: {end_extract - start_extract:.2f} segundos")
    print(f"🕒 Tiempo de extracción: {end_extract - start_extract:.2f} segundos")
    logging.info(f"Tamaño de X_train_feats después de extracción: {len(X_train_feats)}")
    print(f"Tamaño de X_train_feats después de extracción: {len(X_train_feats)}")

    logging.info("Extrayendo características para el conjunto de prueba...")
    print("\nExtrayendo características para el conjunto de prueba...")
    start_extract = time.time()
    X_test_feats = extraer_features(X_test)
    if X_test_feats is None:
        logging.error("No se encontraron características para el conjunto de prueba.")
        print("Error: No se encontraron características para el conjunto de prueba.")
        exit(1)
    end_extract = time.time()
    logging.info(f"Tiempo de extracción: {end_extract - start_extract:.2f} segundos")
    print(f"🕒 Tiempo de extracción: {end_extract - start_extract:.2f} segundos")

    logging.info("Extrayendo características para el conjunto de validación...")
    print("\nExtrayendo características para el conjunto de validación...")
    start_extract = time.time()
    X_val_feats = extraer_features(X_val)
    if X_val_feats is None:
        logging.error("No se pudieron extraer características para el conjunto de validación.")
        print("Error: No se pudieron extraer características para el conjunto de validación.")
        exit(1)
    end_extract = time.time()
    logging.info(f"Tiempo de extracción: {end_extract - start_extract:.2f} segundos")
    print(f"🕒 Tiempo de extracción: {end_extract - start_extract:.2f} segundos")

    logging.info(f"Número de características extraídas: {len(X_train_feats.columns)}")
    print(f"\nNúmero de características extraídas: {len(X_train_feats.columns)}")
    logging.info(f"Valores faltantes por característica:\n{X_train_feats.isna().sum().to_dict()}")
    print(f"\nValores faltantes por característica:\n{X_train_feats.isna().sum()}")

    constant_features = X_train_feats.columns[X_train_feats.nunique() == 1]
    logging.info(f"Características constantes: {constant_features.tolist()}")
    print("\nCaracterísticas constantes:", constant_features)
    if len(constant_features) > 0:
        X_train_feats = X_train_feats.drop(columns=constant_features)
        X_test_feats = X_test_feats.drop(columns=constant_features)
        X_val_feats = X_val_feats.drop(columns=constant_features)
        numeric_features = [f for f in numeric_features if f not in constant_features]
        categorical_features = [f for f in categorical_features if f not in constant_features]
        logging.info(f"Características constantes eliminadas: {len(constant_features)}")
        print(f"Características constantes eliminadas: {len(constant_features)}")

    logging.info(f"Columnas en X_train_feats después de eliminar constantes: {X_train_feats.columns.tolist()}")
    print("\nColumnas en X_train_feats después de eliminar constantes:", X_train_feats.columns.tolist())
    logging.info(f"Características categóricas actualizadas: {categorical_features}")
    print("Características categóricas actualizadas:", categorical_features)

    return X_train_feats, X_test_feats, X_val_feats, numeric_features, categorical_features

def train_and_evaluate(pipelines, X_train_feats, y_train, X_test_feats, y_test, X_val_feats, y_val, df_train, df_test, df_val, numeric_features, categorical_features, output_dir):
    logging.info(f"Resultados se guardarán en '{output_dir}'.")
    print(f"Resultados se guardarán en '{output_dir}'.")
    
    param_dist = {}
            
    models = {}
    roc_data = {'test': {}, 'val': {}}
    f1_scores = {'test': {}, 'val': {}}
    training_times = {}  # Diccionario para almacenar los tiempos de entrenamiento

    for model_name, pipeline in pipelines.items():
        logging.info(f"Entrenando {model_name}...")
        print(f"\n=== Entrenando {model_name} ===")
        import time
        start_train = time.time()
        
        try:
            if GENERAL_PARAMS['ENABLE_HYPERPARAM_SEARCH']:
                if model_name == 'VotingClassifier':
                    param_dist.update(HYPERPARAM_RANGES_UPDATED['VotingClassifier'])
                else:
                    param_dist.update(HYPERPARAM_RANGES_UPDATED[model_name])

                best_model, best_params, best_score = perform_hyperparameter_search(
                    pipeline, param_dist, X_train_feats, y_train, search_type='wide'
                )
                logging.info(f"Ajustando el mejor modelo para {model_name}...")
                print(f"Ajustando el mejor modelo para {model_name}...")
                best_model.fit(X_train_feats, y_train)
            else:
                logging.info(f"Usando pipeline predeterminado para {model_name} (sin búsqueda de hiperparámetros)...")
                print(f"Usando pipeline predeterminado para {model_name} (sin búsqueda de hiperparámetros)...")
                best_model = pipeline
                best_model.fit(X_train_feats, y_train)
            end_train = time.time()
            logging.info(f"Tiempo de entrenamiento: {end_train - start_train:.2f} segundos")
            print(f"🕒 Tiempo de entrenamiento: {end_train - start_train:.2f} segundos")
            training_times[model_name] = end_train - start_train
            models[model_name] = best_model

            logging.info(f"Estado del preprocesador para {model_name}...")
            print(f"\n🔍 Estado del preprocesador para {model_name}...")
            preprocessor = best_model.named_steps['preprocessor']
            for name, transformer, columns in preprocessor.transformers_[:-1]:
                logging.info(f"Preprocesador {name}: {transformer}")
                print(f"Preprocesador {name}: {transformer}")
                try:
                    check_is_fitted(transformer)
                    logging.info(f"Preprocesador {name} está ajustado.")
                    print(f"Preprocesador {name} está ajustado.")
                except:
                    logging.warning(f"Preprocesador {name} NO está ajustado.")
                    print(f"Preprocesador {name} NO está ajustado.")

            logging.info(f"Generando nombres de características para {model_name}...")
            print(f"\n🔍 Generando nombres de características para {model_name}...")
            feature_names = []
            for name, transformer, columns in preprocessor.transformers_[:-1]:
                try:
                    if name == 'num':
                        select_kbest = transformer.named_steps.get('select')
                        if select_kbest:
                            check_is_fitted(select_kbest)
                            mask = select_kbest.get_support()
                            selected_features = [columns[i] for i in range(len(columns)) if mask[i]]
                            feature_names.extend(selected_features)
                        else:
                            feature_names.extend(columns)
                    elif name == 'cat':
                        check_is_fitted(transformer)
                        feature_names.extend(transformer.named_steps['onehot'].get_feature_names_out())
                    elif name in ['text_full', 'text_diff']:
                        n_components = transformer.named_steps['svd'].n_components
                        feature_names.extend([f'{name}_component_{i+1}' for i in range(n_components)])
                except Exception as e:
                    logging.error(f"Error al procesar Preprocesador {name}: {e}")
                    print(f"Error al procesar Preprocesador {name}: {e}")
                    feature_names.extend(columns if name in ['num', 'cat'] else [f'{name}_component_{i+1}' for i in range(100)])

            for dataset_name, X, y in [('test', X_test_feats, y_test), ('val', X_val_feats, y_val)]:
                logging.info(f"Evaluación en Conjunto {dataset_name.capitalize()} ({model_name})...")
                print(f"\n=== Evaluación en Conjunto {dataset_name.capitalize()} ({model_name}) ===")
                start_pred = time.time()
                y_pred = best_model.predict(X)
                try:
                    y_prob = best_model.predict_proba(X)[:, 1]
                except AttributeError:
                    y_prob = best_model.decision_function(X) if hasattr(best_model, 'decision_function') else y_pred
                end_pred = time.time()
                logging.info(f"Tiempo de predicción: {end_pred - start_pred:.2f} segundos")
                print(f"🕒 Tiempo de predicción: {end_pred - start_pred:.2f} segundos")

                umbral_mcc, mcc_val = encontrar_umbral_optimo_mcc(y, y_prob)
                logging.info(f"Umbral óptimo (MCC): {umbral_mcc:.3f}, MCC: {mcc_val:.3f}")
                print(f"🔍 Umbral óptimo (MCC): {umbral_mcc:.3f}, MCC: {mcc_val:.3f}")
                y_pred_mcc = (y_prob >= umbral_mcc).astype(int)

                logging.info(f"Reporte de Clasificación (umbral óptimo MCC, {model_name}):")
                logging.info(classification_report(y, y_pred_mcc))
                print(f"\n📊 Reporte de Clasificación (umbral óptimo MCC, {model_name}):")
                print(classification_report(y, y_pred_mcc))
                logging.info(f"Matriz de Confusión (umbral óptimo MCC, {model_name}):")
                logging.info(str(confusion_matrix(y, y_pred_mcc)))
                print(f"Matriz de Confusión (umbral óptimo MCC, {model_name}):")
                print(confusion_matrix(y, y_pred_mcc))

                fpr, tpr, _ = roc_curve(y, y_prob)
                roc_auc = roc_auc_score(y, y_prob)
                roc_data[dataset_name][model_name] = (fpr, tpr, roc_auc)
                f1_scores[dataset_name][model_name] = classification_report(y, y_pred_mcc, output_dict=True)['1']['f1-score']
                logging.info(f"Área bajo la curva ROC: {roc_auc:.4f}")
                print(f"Área bajo la curva ROC: {roc_auc:.4f}")

            logging.info(f"Calculando Permutation Feature Importance ({model_name})...")
            print(f"\n🔍 Calculando Permutation Feature Importance ({model_name})...")
            sample_size = min(2000, len(X_test_feats))
            sample_idx = np.random.choice(len(X_test_feats), size=sample_size, replace=False)

            try:
                result = permutation_importance(
                    best_model,
                    X_test_feats.iloc[sample_idx],
                    y_test.iloc[sample_idx],
                    n_repeats=10,
                    random_state=GENERAL_PARAMS['RANDOM_STATE'],
                    scoring='f1_macro',
                    n_jobs= 2  # Usar todos los núcleos disponibles
                )

                n_features = len(result.importances_mean)
                importancia_perm = pd.DataFrame({
                    'Feature': feature_names[:n_features],
                    'Importance Mean': result.importances_mean,
                    'Importance Std': result.importances_std
                }).sort_values(by='Importance Mean', ascending=False)

                logging.info(f"Top 20 características más importantes (Permutation Importance, {model_name}):")
                logging.info(importancia_perm.head(20).to_string(index=False))
                print(f"\n📊 Top 20 características más importantes (Permutation Importance, {model_name}):")
                print(importancia_perm.head(20).to_string(index=False))
                importancia_perm.to_csv(os.path.join(output_dir, f"feature_importance_{model_name}.csv"), index=False)

                plt.figure(figsize=(10, 8))
                plt.barh(importancia_perm['Feature'].head(20)[::-1],
                         importancia_perm['Importance Mean'].head(20)[::-1],
                         xerr=importancia_perm['Importance Std'].head(20)[::-1],
                         color='steelblue')
                plt.xlabel("Importancia promedio (F1-macro)")
                plt.title(f"Top 20 Características por Permutation Importance ({model_name})")
                plt.grid(True)
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, f"feature_importance_{model_name}.png"))
                plt.close()
            except Exception as e:
                logging.error(f"Error al calcular Permutation Feature Importance para {model_name}: {e}")
                print(f"Error al calcular Permutation Feature Importance para {model_name}: {e}")
                importancia_perm = pd.DataFrame({'Feature': [], 'Importance Mean': [], 'Importance Std': []})
                importancia_perm.to_csv(os.path.join(output_dir, f"feature_importance_{model_name}.csv"), index=False)

            model_filename = os.path.join(output_dir, f"{model_name.lower()}_pipeline_model.joblib")
            joblib.dump(best_model, model_filename)
            logging.info(f"Modelo {model_name} guardado en '{model_filename}'.")
            print(f"\n✅ Modelo {model_name} guardado en '{model_filename}'.")
            
        except (OverflowError, MemoryError) as e:
            logging.error(f"Error crítico en {model_name}: {e}", exc_info=True)
            print(f"❌ Error crítico en {model_name}: {e}")
            import traceback
            traceback.print_exc()
            logging.warning(f"Continuando con el siguiente modelo.")
            continue
        except Exception as e:
            logging.error(f"Error inesperado en {model_name}: {e}", exc_info=True)
            print(f"❌ Error inesperado en {model_name}: {e}")
            import traceback
            traceback.print_exc()
            logging.warning(f"Deteniendo la ejecución.")
            raise

    plt.figure(figsize=(10, 8))
    for model_name, (fpr, tpr, roc_auc) in roc_data['test'].items():
        plt.plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel("Tasa de Falsos Positivos")
    plt.ylabel("Tasa de Verdaderos Positivos")
    plt.title("Curvas ROC (Conjunto de Prueba)")
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "roc_test.png"))
    plt.close()

    plt.figure(figsize=(10, 8))
    for model_name, (fpr_val, tpr_val, roc_auc_val) in roc_data['val'].items():
        plt.plot(fpr_val, tpr_val, label=f'{model_name} (AUC = {roc_auc_val:.4f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel("Tasa de Falsos Positivos")
    plt.ylabel("Tasa de Verdaderos Positivos")
    plt.title("Curvas ROC (Conjunto de Validación)")
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "roc_val.png"))
    plt.close()

    # Generar gráfica de barras
    plt.figure(figsize=(10, 6))
    model_names = list(training_times.keys())
    times = list(training_times.values())
    plt.bar(model_names, times, color=['#1f77b4', '#ff7f0e', '#2ca02c'], edgecolor='black')
    plt.xlabel("Modelo")
    plt.ylabel("Tiempo de Entrenamiento (segundos)")
    plt.title("Comparación de Tiempos de Entrenamiento por Modelo")
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    for i, time in enumerate(times):
        plt.text(i, time + 0.5, f"{time:.2f}s", ha='center', va='bottom')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "training_times_comparison.png"))
    plt.close()
    logging.info(f"Gráfica de tiempos de entrenamiento guardada en '{output_dir}/training_times_comparison.png'")
    print(f"Gráfica de tiempos de entrenamiento guardada en '{output_dir}/training_times_comparison.png'")

    logging.info("Comparación de F1-scores (Clase 1)...")
    print("\n=== Comparación de F1-scores (Clase 1) ===")
    f1_df = pd.DataFrame({
        'Test': f1_scores['test'],
        'Validation': f1_scores['val']
    })
    logging.info(f1_df.to_string())
    print(f1_df.to_string())
    f1_df.to_csv(os.path.join(output_dir, "f1_scores.csv"), index=True)

    if 'VotingClassifier' in models:
        best_pipeline = models['VotingClassifier']
        logging.info("Evaluación Adicional para VotingClassifier...")
        print("\n=== Evaluación Adicional para VotingClassifier ===")
        y_pred = best_pipeline.predict(X_test_feats)
        try:
            y_prob = best_pipeline.predict_proba(X_test_feats)[:, 1]
        except AttributeError:
            y_prob = y_pred

        precisions, recalls, thresholds_pr = precision_recall_curve(y_test, y_prob)
        f1_scores_pr = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
        optimal_idx = np.argmax(f1_scores_pr)
        optimal_threshold = thresholds_pr[optimal_idx]
        logging.info(f"Umbral óptimo (máximo F1-score): {optimal_threshold:.4f}")
        print(f"\nUmbral óptimo (máximo F1-score): {optimal_threshold:.4f}")
        y_pred_optimal = (y_prob >= optimal_threshold).astype(int)
        logging.info(f"Reporte de Clasificación (umbral={optimal_threshold:.4f}):")
        logging.info(classification_report(y_test, y_pred_optimal))
        print(f"\nReporte de Clasificación (umbral={optimal_threshold:.4f}):")
        print(classification_report(y_test, y_pred_optimal))
        logging.info(f"Matriz de Confusión (umbral óptimo):")
        logging.info(str(confusion_matrix(y_test, y_pred_optimal)))
        print("\nMatriz de Confusión (umbral óptimo):")
        print(confusion_matrix(y_test, y_pred_optimal))

        logging.info("Distribución de Probabilidades Predichas (Clase 1, VotingClassifier):")
        prob_stats = pd.Series(y_prob).describe()
        logging.info(prob_stats.to_string())
        print("\n📈 Distribución de Probabilidades Predichas (Clase 1, VotingClassifier):")
        print(prob_stats.to_string())
        plt.figure(figsize=(8, 6))
        plt.hist(y_prob, bins=20, density=True, color='skyblue', edgecolor='black')
        plt.title("Distribución de Probabilidades Predichas (Clase 1, VotingClassifier)")
        plt.xlabel("Probabilidad")
        plt.ylabel("Densidad")
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, "prob_dist.png"))
        plt.close()

        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(6, 5))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title("Matriz de Confusión (Prueba, VotingClassifier, umbral=0.5)")
        plt.colorbar()
        tick_marks = np.arange(2)
        plt.xticks(tick_marks, ["No Defecto", "Defecto"])
        plt.yticks(tick_marks, ["No Defecto", "Defecto"])
        thresh = cm.max() / 2
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, format(cm[i, j], 'd'),
                         horizontalalignment="center",
                         color="white" if cm[i, j] > thresh else "black")
        plt.ylabel("Etiqueta Verdadera")
        plt.xlabel("Etiqueta Predicha")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "confusion_matrix.png"))
        plt.close()

        logging.info("Métricas por Repositorio (Prueba, VotingClassifier)...")
        print("\n=== Métricas por Repositorio (Prueba, VotingClassifier) ===")
        df_test['pred'] = y_pred

        def get_f1_score(group):
            repo_name = group['repo'].iloc[0]
            try:
                if 1 not in group['label'].values and 1 not in group['pred'].values:
                    logging.info(f"Repositorio '{repo_name}' sin clase '1' (muestras: {len(group)})")
                    return 0
                report = classification_report(group['label'], group['pred'], output_dict=True, zero_division=0)
                return report.get('1', {'f1-score': 0})['f1-score']
            except ValueError as e:
                logging.error(f"Error en repositorio '{repo_name}': {e} (muestras: {len(group)})")
                return 0

        repo_metrics = df_test.groupby('repo').apply(get_f1_score).sort_values(ascending=False)
        logging.info("F1-score para clase 1 por repositorio:")
        logging.info(repo_metrics.to_string())
        print("F1-score para clase 1 por repositorio:")
        print(repo_metrics.to_string())
        repo_metrics.to_csv(os.path.join(output_dir, "repo_metrics.csv"), index=True)

        logging.info("Ejemplos de Errores de Predicción (Prueba, VotingClassifier)...")
        print("\n=== Ejemplos de Errores de Predicción (Prueba, VotingClassifier) ===")
        fp = df_test[(df_test['label'] == 0) & (df_test['pred'] == 1)][['repo', 'filepath', 'content_diff', 'content_full']].head(3)
        fn = df_test[(df_test['label'] == 1) & (df_test['pred'] == 0)][['repo', 'filepath', 'content_diff', 'content_full']].head(3)
        fp['content_diff'] = fp['content_diff'].str[:200]
        fp['content_full'] = fp['content_full'].str[:200]
        fn['content_diff'] = fn['content_diff'].str[:200]
        fn['content_full'] = fn['content_full'].str[:200]
        logging.info("Falsos Positivos (predicho como defecto, pero no lo es):")
        logging.info(fp.to_string())
        print("\nFalsos Positivos (predicho como defecto, pero no lo es):")
        print(fp.to_string())
        fp.to_csv(os.path.join(output_dir, "false_positives.csv"), index=False)
        logging.info("Falsos Negativos (defecto no detectado):")
        logging.info(fn.to_string())
        print("\nFalsos Negativos (defecto no detectado):")
        print(fn.to_string())
        fn.to_csv(os.path.join(output_dir, "false_negatives.csv"), index=False)

    logging.info("Análisis de Repositorios y Nomenclatura de Archivos...")
    print("\n=== Análisis de Repositorios y Nomenclatura de Archivos ===")
    repo_stats = df_train.groupby('repo').agg({
        'repo': ['count'],
        'filepath': ['nunique']
    })
    repo_stats.columns = ['n_commits', 'n_unique_files']
    repo_stats = repo_stats.reset_index()
    repo_stats = repo_stats.merge(repo_metrics.rename('f1_score'), on='repo', how='left').fillna({'f1_score': 0})
    logging.info("Top 10 repositorios por cantidad de commits con F1-score:")
    logging.info(repo_stats.sort_values(by='n_commits', ascending=False)[['repo', 'n_commits', 'n_unique_files', 'f1_score']].head(10).to_string(index=False))
    print("\nTop 10 repositorios por cantidad de commits con F1-score:")
    print(repo_stats.sort_values(by='n_commits', ascending=False)[['repo', 'n_commits', 'n_unique_files', 'f1_score']].head(10).to_string(index=False))
    repo_stats.to_csv(os.path.join(output_dir, "repo_stats.csv"), index=False)

    df_train['file_extension'] = df_train['filepath'].apply(lambda x: x.split('.')[-1] if '.' in x else 'none')
    extension_stats = df_train.groupby('repo')['file_extension'].value_counts(normalize=True).unstack().fillna(0)
    logging.info("Distribución de extensiones de archivo en algunos repositorios:")
    logging.info(extension_stats.head(10).to_string())
    print("\nDistribución de extensiones de archivo en algunos repositorios:")
    print(extension_stats.head(10).to_string())
    extension_stats.to_csv(os.path.join(output_dir, "extension_stats.csv"), index=True)

    file_naming_stats = df_train.groupby('repo').agg(
        avg_filepath_length=('filepath', lambda x: x.str.len().mean())
    ).reset_index()
    logging.info("Longitud promedio de nombres de archivos por repositorio (top 10):")
    logging.info(file_naming_stats.sort_values(by='avg_filepath_length', ascending=False).head(10).to_string(index=False))
    print("\nLongitud promedio de nombres de archivos por repositorio (top 10):")
    print(file_naming_stats.sort_values(by='avg_filepath_length', ascending=False).head(10).to_string(index=False))
    file_naming_stats.to_csv(os.path.join(output_dir, "file_naming_stats.csv"), index=True)

    plt.figure(figsize=(8, 6))
    plt.scatter(repo_stats['n_commits'], repo_stats['f1_score'], alpha=0.5)
    plt.xlabel("Número de Commits")
    plt.ylabel("F1-score Clase '1'")
    plt.title("Commits vs. F1-score por Repositorio")
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "repo_scatter.png"))
    plt.close()

    logging.info("Análisis de Defectos por Hora y Día...")
    print("\n=== Análisis de Defectos por Hora y Día ===")
    try:
        df_train['commit_datetime'] = pd.to_datetime(df_train['datetime'], utc=True)
        df_train['commit_hour'] = df_train['commit_datetime'].dt.hour
        df_train['commit_day'] = df_train['commit_datetime'].dt.dayofweek
        logging.info("Distribución de defectos por hora:")
        logging.info(df_train.groupby('commit_hour')['label'].mean().to_string())
        print("\nDistribución de defectos por hora:")
        print(df_train.groupby('commit_hour')['label'].mean())
        logging.info("Distribución de defectos por día:")
        logging.info(df_train.groupby('commit_day')['label'].mean().to_string())
        print("\nDistribución de defectos por día:")
        print(df_train.groupby('commit_day')['label'].mean())
    except Exception as e:
        logging.error(f"Error al analizar defectos por hora y día: {e}")
        print(f"Error al analizar defectos por hora y día: {e}")

    logging.info("Proceso completado.")
    print("\n🎉 Proceso completado.")