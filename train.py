import pandas as pd
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Usar backend no interactivo para evitar conflictos con tkinter
import matplotlib.pyplot as plt
import joblib
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, RandomizedSearchCV
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, roc_auc_score, precision_recall_curve
from sklearn.inspection import permutation_importance
from features import extraer_features
from utils import encontrar_umbral_optimo_mcc
from config import GENERAL_PARAMS

def load_data():
    print("Cargando los datasets...")
    # Cargar datasets Line
    df_train_line = pd.read_parquet("line_bug_prediction_splits/random/train.parquet.gzip")
    df_test_line = pd.read_parquet("line_bug_prediction_splits/random/test.parquet.gzip")
    df_val_line = pd.read_parquet("line_bug_prediction_splits/random/val.parquet.gzip")
    # Cargar datasets JIT
    df_train_jit = pd.read_parquet("jit_bug_prediction_splits/random/train.parquet.gzip")
    df_test_jit = pd.read_parquet("jit_bug_prediction_splits/random/test.parquet.gzip")
    df_val_jit = pd.read_parquet("jit_bug_prediction_splits/random/val.parquet.gzip")
    print("Columnas en df_train_line:", df_train_line.columns.tolist())
    print("Columnas en df_train_jit:", df_train_jit.columns.tolist())
    print("Datasets cargados.")
    return (df_train_line, df_test_line, df_val_line), (df_train_jit, df_test_jit, df_val_jit)

def process_data(dfs_line, dfs_jit, data_type):
    df_train_line, df_test_line, df_val_line = dfs_line
    df_train_jit, df_test_jit, df_val_jit = dfs_jit
    
    # Fusionar JIT y Line
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
        print("\nTipo 0: Usar datasets originales sin modificar ni estratificar")
        df_train['label'] = df_train['lines'].apply(lambda x: 0 if len(x) == 0 else 1)
        df_test['label'] = df_test['lines'].apply(lambda x: 0 if len(x) == 0 else 1)
        df_val['label'] = df_val['lines'].apply(lambda x: 0 if len(x) == 0 else 1)
    elif data_type == 1:
        print("\nTipo 1: Combinar, estratificar y muestrear")
        data = pd.concat([df_train, df_test, df_val]).reset_index(drop=True)
        data['label'] = data['lines'].apply(lambda x: 0 if len(x) == 0 else 1)
        print("\nDistribución de clases original (antes de muestreo):")
        print(data['label'].value_counts(normalize=True).to_string())
        print("\nPreparando conjuntos de datos con estratificación...")
        X = data.drop(columns=['label'])
        y = data['label']
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=GENERAL_PARAMS['RANDOM_STATE']
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=0.25, stratify=y_temp, random_state=GENERAL_PARAMS['RANDOM_STATE']
        )
        X_train = X_train.sample(frac=GENERAL_PARAMS['DATA_FRACTION'], random_state=GENERAL_PARAMS['RANDOM_STATE'])
        y_train = y_train.loc[X_train.index]
        X_test = X_test.sample(frac=GENERAL_PARAMS['DATA_FRACTION'], random_state=GENERAL_PARAMS['RANDOM_STATE'])
        y_test = y_test.loc[X_test.index]
        X_val = X_val.sample(frac=GENERAL_PARAMS['DATA_FRACTION'], random_state=GENERAL_PARAMS['RANDOM_STATE'])
        y_val = y_val.loc[X_val.index]
        df_train = X_train.copy()
        df_train['label'] = y_train
        df_test = X_test.copy()
        df_test['label'] = y_test
        df_val = X_val.copy()
        df_val['label'] = y_val
    else:
        raise ValueError("data_type debe ser 0 o 1")

    # Preprocesar contenido
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

    print(f"Tamaño del conjunto de entrenamiento: {len(df_train)}")
    print(f"Tamaño del conjunto de prueba: {len(df_test)}")
    print(f"Tamaño del conjunto de validación: {len(df_val)}")
    print("Conjuntos de datos preparados.")
    
    return X_train, y_train, X_test, y_test, X_val, y_val, df_train, df_test, df_val

def extract_features(X_train, X_test, X_val, numeric_features, categorical_features):
    print("\nExtrayendo características para el conjunto de entrenamiento...")
    start_extract = time.time()
    X_train_feats = extraer_features(X_train)
    if X_train_feats is None:
        print("Error: No se pudieron extraer características para el conjunto de entrenamiento.")
        exit(1)
    end_extract = time.time()
    print(f"🕒 Tiempo de extracción: {end_extract - start_extract:.2f} segundos")
    print(f"Tamaño de X_train_feats después de extracción: {len(X_train_feats)}")

    print("\nExtrayendo características para el conjunto de prueba...")
    start_extract = time.time()
    X_test_feats = extraer_features(X_test)
    if X_test_feats is None:
        print("Error: No se pudieron extraer características para el conjunto de prueba.")
        exit(1)
    end_extract = time.time()
    print(f"🕒 Tiempo de extracción: {end_extract - start_extract:.2f} segundos")

    print("\nExtrayendo características para el conjunto de validación...")
    start_extract = time.time()
    X_val_feats = extraer_features(X_val)
    if X_val_feats is None:
        print("Error: No se pudieron extraer características para el conjunto de validación.")
        exit(1)
    end_extract = time.time()
    print(f"🕒 Tiempo de extracción: {end_extract - start_extract:.2f} segundos")

    print(f"\nNúmero de características extraídas: {len(X_train_feats.columns)}")
    print(f"Valores faltantes por característica:\n{X_train_feats.isna().sum()}")

    constant_features = X_train_feats.columns[X_train_feats.nunique() == 1]
    print("\nCaracterísticas constantes:", constant_features)
    if len(constant_features) > 0:
        X_train_feats = X_train_feats.drop(columns=constant_features)
        X_test_feats = X_test_feats.drop(columns=constant_features)
        X_val_feats = X_val_feats.drop(columns=constant_features)
        numeric_features = [f for f in numeric_features if f not in constant_features]
        categorical_features = [f for f in categorical_features if f not in constant_features]
        print(f"Características constantes eliminadas: {len(constant_features)}")

    print("\nColumnas en X_train_feats después de eliminar constantes:", X_train_feats.columns.tolist())
    print("Características categóricas actualizadas:", categorical_features)

    return X_train_feats, X_test_feats, X_val_feats, numeric_features, categorical_features

def train_and_evaluate(pipelines, X_train_feats, y_train, X_test_feats, y_test, X_val_feats, y_val, df_train, df_test, df_val, numeric_features, categorical_features):
    param_dist = {
        # RandomForestClassifier
        'clf__estimator__n_estimators': [100, 200, 500],
        'clf__estimator__max_depth': [None, 10, 20, 30],
        'clf__estimator__min_samples_leaf': [1, 2, 4],
        'clf__estimator__min_samples_split': [2, 5, 10],
        'clf__estimator__max_features': ['sqrt', 'log2', None],
        'clf__estimator__class_weight': [None, 'balanced', 'balanced_subsample'],
        # XGBClassifier
        'clf__estimator__n_estimators': [100, 200, 300],
        'clf__estimator__learning_rate': [0.01, 0.05, 0.1, 0.2],
        'clf__estimator__max_depth': [3, 5, 7, 10],
        'clf__estimator__subsample': [0.6, 0.8, 1.0],
        'clf__estimator__colsample_bytree': [0.6, 0.8, 1.0],
        # LinearSVC
        'clf__estimator__C': [0.01, 0.1, 1.0, 10.0],
        # TF-IDF + SVD
        'preprocessor__text_full__tfidf__max_df': [0.7, 0.85, 1.0],
        'preprocessor__text_full__tfidf__min_df': [1, 3, 5],
        'preprocessor__text_full__svd__n_components': [20, 50, 100],
        'preprocessor__text_diff__tfidf__max_df': [0.7, 0.85, 1.0],
        'preprocessor__text_diff__tfidf__min_df': [1, 3, 5],
        'preprocessor__text_diff__svd__n_components': [20, 50, 100]
    }

    models = {}
    roc_data = {'test': {}, 'val': {}}
    f1_scores = {'test': {}, 'val': {}}

    for model_name, pipeline in pipelines.items():
        print(f"\n=== Entrenando {model_name} ===")
        pipeline.set_params(
            preprocessor__text_full__tfidf__min_df=3,
            preprocessor__text_full__tfidf__max_df=1.0,
            preprocessor__text_full__svd__n_components=100,
            preprocessor__text_diff__tfidf__min_df=3,
            preprocessor__text_diff__tfidf__max_df=1.0,
            preprocessor__text_diff__svd__n_components=100,
        )

        start_train = time.time()
        pipeline.fit(X_train_feats, y_train)
        end_train = time.time()
        print(f"🕒 Tiempo de entrenamiento: {end_train - start_train:.2f} segundos")
        models[model_name] = pipeline

        #print(f"\n=== Validación Cruzada ({model_name}) ===")
        #start_pred = time.time()
        #cv_scores = cross_val_score(
        #    pipeline,
        #    X_train_feats,
        #    y_train,
        #    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        #    scoring='f1_macro'
        #)
        #print(f"Puntuaciones F1-macro (5-fold CV): {cv_scores}")
        #print(f"Media: {cv_scores.mean():.4f} (± {cv_scores.std():.4f})")
        #end_pred = time.time()
        #print(f"🕒 Tiempo de Validación Cruzada: {end_pred - start_pred:.2f} segundos")

        print(f"\n=== Evaluación en Conjunto de Prueba ({model_name}) ===")
        start_pred = time.time()
        y_pred = pipeline.predict(X_test_feats)
        y_prob = pipeline.predict_proba(X_test_feats)[:, 1]
        end_pred = time.time()
        print(f"🕒 Tiempo de predicción: {end_pred - start_pred:.2f} segundos")

        umbral_mcc, mcc_val = encontrar_umbral_optimo_mcc(y_test, y_prob)
        print(f"🔍 Umbral óptimo (MCC): {umbral_mcc:.3f}, MCC: {mcc_val:.3f}")
        y_pred_mcc = (y_prob >= umbral_mcc).astype(int)

        print(f"\n📊 Reporte de Clasificación (umbral óptimo MCC, {model_name}):")
        print(classification_report(y_test, y_pred_mcc))
        print(f"Matriz de Confusión (umbral óptimo MCC, {model_name}):")
        print(confusion_matrix(y_test, y_pred_mcc))

        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = roc_auc_score(y_test, y_prob)
        roc_data['test'][model_name] = (fpr, tpr, roc_auc)
        f1_scores['test'][model_name] = classification_report(y_test, y_pred_mcc, output_dict=True)['1']['f1-score']
        print(f"Área bajo la curva ROC: {roc_auc:.4f}")

        print(f"\n=== Evaluación en Conjunto de Validación ({model_name}) ===")
        start_pred = time.time()
        y_val_pred = pipeline.predict(X_val_feats)
        y_val_prob = pipeline.predict_proba(X_val_feats)[:, 1]
        end_pred = time.time()
        print(f"🕒 Tiempo de predicción: {end_pred - start_pred:.2f} segundos")

        print(f"\n📊 Reporte de Clasificación (umbral=0.5, {model_name}):")
        print(classification_report(y_val, y_val_pred))
        print(f"Matriz de Confusión (umbral=0.5, {model_name}):")
        print(confusion_matrix(y_val, y_val_pred))

        fpr_val, tpr_val, _ = roc_curve(y_val, y_val_prob)
        roc_auc_val = roc_auc_score(y_val, y_val_prob)
        roc_data['val'][model_name] = (fpr_val, tpr_val, roc_auc_val)
        f1_scores['val'][model_name] = classification_report(y_val, y_val_pred, output_dict=True)['1']['f1-score']
        print(f"Área bajo la curva ROC: {roc_auc_val:.4f}")

    # Visualización Comparativa de Curvas ROC
    plt.figure(figsize=(10, 8))
    for model_name, (fpr, tpr, roc_auc) in roc_data['test'].items():
        plt.plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel("Tasa de Falsos Positivos")
    plt.ylabel("Tasa de Verdaderos Positivos")
    plt.title("Curvas ROC (Conjunto de Prueba)")
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.savefig("roc_test.png")
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
    plt.savefig("roc_val.png")
    plt.close()

    # Comparación de F1-scores
    print("\n=== Comparación de F1-scores (Clase 1) ===")
    f1_df = pd.DataFrame({
        'Test': f1_scores['test'],
        'Validation': f1_scores['val']
    })
    print(f1_df.to_string())
    f1_df.to_csv("f1_scores.csv", index=True)

    # Evaluación Adicional para VotingClassifier
    best_pipeline = models['VotingClassifier']
    print("\n=== Evaluación Adicional para VotingClassifier ===")

    y_pred = best_pipeline.predict(X_test_feats)
    y_prob = best_pipeline.predict_proba(X_test_feats)[:, 1]

    precisions, recalls, thresholds_pr = precision_recall_curve(y_test, y_prob)
    f1_scores_pr = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
    optimal_idx = np.argmax(f1_scores_pr)
    optimal_threshold = thresholds_pr[optimal_idx]
    print(f"Umbral óptimo (máximo F1-score): {optimal_threshold:.4f}")
    y_pred_optimal = (y_prob >= optimal_threshold).astype(int)
    print(f"Reporte de Clasificación (umbral={optimal_threshold:.4f}):")
    print(classification_report(y_test, y_pred_optimal))
    print("Matriz de Confusión (umbral óptimo):")
    print(confusion_matrix(y_test, y_pred_optimal))

    print("\n📈 Distribución de Probabilidades Predichas (Clase 1, VotingClassifier):")
    prob_stats = pd.Series(y_prob).describe()
    print(prob_stats.to_string())
    plt.figure(figsize=(8, 6))
    plt.hist(y_prob, bins=20, density=True, color='skyblue', edgecolor='black')
    plt.title("Distribución de Probabilidades Predichas (Clase 1, VotingClassifier)")
    plt.xlabel("Probabilidad")
    plt.ylabel("Densidad")
    plt.grid(True)
    plt.savefig("prob_dist.png")
    plt.close()

    # Matriz de Confusión (Prueba, VotingClassifier)
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Matriz de Confusión (Prueba, VotingClassifier, umbral=0.5)")
    plt.colorbar()
    tick_marks = range(2)
    plt.xticks(tick_marks, ["No Defecto", "Defecto"])
    plt.yticks(tick_marks, ["No Defecto", "Defecto"])
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
    plt.ylabel("Etiqueta Verdadera")
    plt.xlabel("Etiqueta Predicha")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png")
    plt.close()

    # Permutation Feature Importance (VotingClassifier)
    print("\n🔍 Calculando Permutation Feature Importance (VotingClassifier)...")
    print(f"Tamaño de X_test_feats: {len(X_test_feats)}")
    sample_size = min(5000, len(X_test_feats))
    sample_idx = np.random.choice(len(X_test_feats), size=sample_size, replace=False)

    feature_names = []
    for name, transformer, columns in best_pipeline.named_steps['preprocessor'].transformers_:
        if name == 'num':
            select_kbest = transformer.named_steps.get('select')
            if select_kbest:
                mask = select_kbest.get_support()
                selected_features = [columns[i] for i in range(len(columns)) if mask[i]]
                feature_names.extend(selected_features)
            else:
                feature_names.extend(columns)
        elif name == 'text_full':
            n_components = transformer.named_steps['svd'].n_components
            feature_names.extend([f'text_full_component_{i+1}' for i in range(n_components)])
        elif name == 'text_diff':
            n_components = transformer.named_steps['svd'].n_components
            feature_names.extend([f'text_diff_component_{i+1}' for i in range(n_components)])
        elif name == 'cat':
            feature_names.extend(transformer.get_feature_names_out())

    try:
        result = permutation_importance(
            best_pipeline,
            X_test_feats.iloc[sample_idx],
            y_test.iloc[sample_idx],
            n_repeats=20,
            random_state=42,
            scoring='f1_macro',
            n_jobs=-1  # Usar un solo hilo para evitar conflictos
        )

        n_features = len(result.importances_mean)
        importancia_perm = pd.DataFrame({
            'Feature': feature_names[:n_features],
            'Importance Mean': result.importances_mean,
            'Importance Std': result.importances_std
        }).sort_values(by='Importance Mean', ascending=False)

        print("\n📊 Top 20 características más importantes (Permutation Importance):")
        print(importancia_perm.head(20).to_string(index=False))
        importancia_perm.to_csv("feature_importance.csv", index=False)

        plt.figure(figsize=(10, 6))
        plt.barh(importancia_perm['Feature'].head(20)[::-1],
                 importancia_perm['Importance Mean'].head(20)[::-1],
                 xerr=importancia_perm['Importance Std'].head(20)[::-1],
                 color='steelblue')
        plt.xlabel("Importancia promedio (f1_macro)")
        plt.title("Top 20 características por Permutation Feature Importance")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("feature_importance.png")
        plt.close()
    except Exception as e:
        print(f"Error al calcular Permutation Feature Importance: {e}")
        importancia_perm = pd.DataFrame({'Feature': [], 'Importance Mean': [], 'Importance Std': []})
        importancia_perm.to_csv("feature_importance.csv", index=False)

    # Métricas por Repositorio (VotingClassifier)
    print("\n=== Métricas por Repositorio (Prueba, VotingClassifier) ===")
    df_test['pred'] = y_pred

    def get_f1_score(group):
        repo_name = group['repo'].iloc[0]
        try:
            if 1 not in group['label'].values and 1 not in group['pred'].values:
                print(f"Repositorio '{repo_name}' sin clase '1' (muestras: {len(group)})")
                return 0
            report = classification_report(group['label'], group['pred'], output_dict=True, zero_division=0)
            return report.get('1', {'f1-score': 0})['f1-score']
        except ValueError as e:
            print(f"Error en repositorio '{repo_name}': {e} (muestras: {len(group)})")
            return 0

    repo_metrics = df_test.groupby('repo').apply(get_f1_score).sort_values(ascending=False)
    print("F1-score para clase 1 por repositorio:")
    print(repo_metrics.to_string())
    repo_metrics.to_csv("repo_metrics.csv", index=True)

    # Ejemplos de Errores de Predicción (VotingClassifier)
    print("\n=== Ejemplos de Errores de Predicción (Prueba, VotingClassifier) ===")
    fp = df_test[(df_test['label'] == 0) & (df_test['pred'] == 1)][['repo', 'filepath', 'content_diff', 'content_full']].head(3)
    fn = df_test[(df_test['label'] == 1) & (df_test['pred'] == 0)][['repo', 'filepath', 'content_diff', 'content_full']].head(3)
    fp['content_diff'] = fp['content_diff'].str[:200]
    fp['content_full'] = fp['content_full'].str[:200]
    fn['content_diff'] = fn['content_diff'].str[:200]
    fn['content_full'] = fn['content_full'].str[:200]
    print("\nFalsos Positivos (predicho como defecto, pero no lo es):")
    print(fp.to_string())
    fp.to_csv("false_positives.csv", index=False)
    print("\nFalsos Negativos (defecto no detectado):")
    print(fn.to_string())
    fn.to_csv("false_negatives.csv", index=False)

    # Guardar Modelo (VotingClassifier)
    joblib.dump(best_pipeline, "rf_pipeline_model.joblib")
    print("\n✅ Modelo guardado en 'rf_pipeline_model.joblib'.")

    # Análisis de Repositorios
    print("\n=== Análisis de Repositorios y Nomenclatura de Archivos ===")
    repo_stats = df_train.groupby('repo').agg(
        n_commits=('repo', 'count'),
        n_unique_files=('filepath', 'nunique')
    ).reset_index()
    repo_stats = repo_stats.merge(repo_metrics.rename('f1_score'), on='repo', how='left').fillna({'f1_score': 0})
    print("\nTop 10 repositorios por cantidad de commits con F1-score:")
    print(repo_stats.sort_values(by='n_commits', ascending=False)[['repo', 'n_commits', 'n_unique_files', 'f1_score']].head(10).to_string(index=False))
    repo_stats.to_csv("repo_stats.csv", index=False)

    df_train['file_extension'] = df_train['filepath'].apply(lambda x: x.split('.')[-1] if '.' in x else 'none')
    extension_stats = df_train.groupby('repo')['file_extension'].value_counts(normalize=True).unstack().fillna(0)
    print("\nDistribución de extensiones de archivo en algunos repositorios:")
    print(extension_stats.head(10).to_string())
    extension_stats.to_csv("extension_stats.csv", index=True)

    file_naming_stats = df_train.groupby('repo').agg(
        avg_filepath_length=('filepath', lambda x: x.str.len().mean())
    ).reset_index()
    print("\nLongitud promedio de nombres de archivos por repositorio (top 10):")
    print(file_naming_stats.sort_values(by='avg_filepath_length', ascending=False).head(10).to_string(index=False))
    file_naming_stats.to_csv("file_naming_stats.csv", index=False)

    plt.figure(figsize=(8, 6))
    plt.scatter(repo_stats['n_commits'], repo_stats['f1_score'], alpha=0.5)
    plt.xlabel("Número de Commits")
    plt.ylabel("F1-score Clase '1'")
    plt.title("Commits vs. F1-score por Repositorio")
    plt.grid(True)
    plt.savefig("repo_scatter.png")
    plt.close()

    print("\n✅ Análisis de repositorios completado.")

    # Análisis de defectos por commit_hour y commit_day
    print("\n=== Análisis de Defectos por Hora y Día ===")
    try:
        # Convertir a UTC para manejar zonas horarias
        df_train['commit_datetime'] = pd.to_datetime(df_train['datetime'], utc=True)
        df_train['commit_hour'] = df_train['commit_datetime'].dt.hour
        df_train['commit_day'] = df_train['commit_datetime'].dt.dayofweek
        print("\nDistribución de defectos por hora:")
        print(df_train.groupby('commit_hour')['label'].mean())
        print("\nDistribución de defectos por día:")
        print(df_train.groupby('commit_day')['label'].mean())
    except Exception as e:
        print(f"Error al analizar defectos por hora y día: {e}")