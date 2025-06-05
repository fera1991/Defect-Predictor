from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from config import GENERAL_PARAMS, RF_PARAMS
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif
from config import TFIDF_PARAMS, SVD_PARAMS

def create_preprocessor(numeric_features, categorical_features):
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler()),
        ('var_threshold', VarianceThreshold(threshold=0.0)),
        ('select', SelectKBest(score_func=f_classif, k='all'))
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    text_transformer_full = Pipeline(steps=[
        ('tfidf', TfidfVectorizer(**TFIDF_PARAMS)),
        ('svd', TruncatedSVD(**SVD_PARAMS))
    ])
    
    text_transformer_diff = Pipeline(steps=[
        ('tfidf', TfidfVectorizer(**TFIDF_PARAMS)),
        ('svd', TruncatedSVD(**SVD_PARAMS))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features),
            ('text_full', text_transformer_full, 'content_text_full'),
            ('text_diff', text_transformer_diff, 'content_text_diff')
        ]
    )
    
    return preprocessor

def create_pipelines(preprocessor):
    print("Creando pipelines individuales y VotingClassifier")
    
    clf_rf = RandomForestClassifier(**RF_PARAMS)
    clf_xgb = XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        eval_metric='logloss',
        random_state=GENERAL_PARAMS['RANDOM_STATE'],
        n_jobs=-1
    )
    
    pipeline_rf = ImbPipeline(steps=[
        ('preprocessor', preprocessor),
        ('smote', SMOTE(random_state=GENERAL_PARAMS['RANDOM_STATE'])),
        ('clf', CalibratedClassifierCV(clf_rf, cv=3, method='sigmoid'))
    ])
    
    pipeline_xgb = ImbPipeline(steps=[
        ('preprocessor', preprocessor),
        ('smote', SMOTE(random_state=GENERAL_PARAMS['RANDOM_STATE'])),
        ('clf', CalibratedClassifierCV(clf_xgb, cv=3, method='sigmoid'))
    ])
    

    
    voting_clf = VotingClassifier(
        estimators=[
            ('rf', clf_rf),
            ('xgb', clf_xgb),
            # ('lr', clf_lr)
        ],
        voting='soft',
        weights=[0.4, 0.6]  # Dar más peso a XGBoost
    )
    
    pipeline_voting = ImbPipeline(steps=[
        ('preprocessor', preprocessor),
        ('smote', SMOTE(random_state=GENERAL_PARAMS['RANDOM_STATE'])),
        ('clf', CalibratedClassifierCV(voting_clf, cv=3, method='sigmoid'))
    ])
    
    return {
        'RandomForest': pipeline_rf,
        'XGBoost': pipeline_xgb,
        'VotingClassifier': pipeline_voting
    }