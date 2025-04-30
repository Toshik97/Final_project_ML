import pandas as pd
import dill
import catboost
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from imblearn.pipeline import make_pipeline as make_pipeline_imb
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, mean_absolute_error
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier
from sklearn.compose import ColumnTransformer
from imblearn.pipeline import make_pipeline
import re


def kill_dublicate0(df):  # удаляем явные повторения
    df = df.copy()
    columns_to_check = ['session_id', 'client_id', 'visit_date', 'visit_number', 'utm_source', 'utm_medium',
                        'utm_adcontent', 'utm_keyword', 'target_action']
    duplicates = df.duplicated(subset=columns_to_check, keep='first')
    df = df[~duplicates].copy()
    return df


def remove_duplicates(df):
    df = df.copy()

    columns_to_check = [
        'utm_medium', 'utm_source', 'utm_campaign', 'device_category',
        'device_os', 'device_brand', 'device_browser', 'geo_country',
        'geo_city', 'target_action', 'utm_keyword'
    ]

    duplicates = df.duplicated(subset=columns_to_check, keep='first')
    df = df[~duplicates].copy()

    return df


def extract_and_update_utm(df, page_path_col='hit_page_path'):
    df = df.copy()

    def extract_utm_params(url, param_name):

        if not isinstance(url, str):
            return None

        pattern = re.compile(r"" + re.escape(param_name) + "=([^&]+)")
        match = pattern.search(url)

        if match:
            return match.group(1)
        else:
            return None

    # Определяем параметры для извлечения и столбцы для обновления
    params_to_extract = {
        'utm_campaign_initial': 'utm_campaign',
        'utm_content_initial': 'utm_adcontent',
        'utm_term_initial': 'utm_keyword'
    }

    df[page_path_col] = df[page_path_col].astype(str)

    for initial_param, target_col in params_to_extract.items():
        # Создаем временный столбец
        new_col_name = f'{target_col}_new'
        df[new_col_name] = df[page_path_col].apply(lambda x: extract_utm_params(x, initial_param))

        # Обновляем столбец, только если во временном столбце не NaN
        df[target_col] = df.apply(
            lambda row: row[new_col_name] if pd.notna(row[new_col_name]) else row[target_col],
            axis=1
        )

        # Удаляем временный столбец
        df.drop(columns=[new_col_name], inplace=True)  # inplace=True для изменения DataFrame

    return df


def kill_NaN(df):  # Заменяем пустые значения
    df = df.copy()
    most_values = df[['utm_campaign', 'utm_adcontent', 'utm_keyword', 'utm_source']].mode().iloc[0]
    df[['utm_campaign', 'utm_adcontent', 'utm_keyword', 'utm_source']] = df[
        ['utm_campaign', 'utm_adcontent', 'utm_keyword', 'utm_source']].fillna(most_values)
    df['utm_campaign'] = df['utm_campaign'].apply(lambda x: 'other' if len(x) < 20 else x[:20])

    df = df.drop(columns=['device_model', 'event_value'])

    return df


def kill_nan_device(df):
    df = df.copy()

    most_frequent_brand = df.groupby('device_category')['device_brand'].agg(
        lambda x: x.mode()[0] if not x.mode().empty else 'other')
    df['device_brand'] = df['device_brand'].fillna(df['device_category'].map(most_frequent_brand))

    mask_apple = (df['device_brand'] == 'Apple') & (df['device_os'].isna())
    df.loc[mask_apple, 'device_os'] = 'iOS'
    mask_android = df['device_os'].isna()
    df.loc[mask_android, 'device_os'] = 'Android'

    return df


def filter_data(df):
    columns_to_drop = [
        'session_id',
        'client_id',
        'visit_date',
        'visit_time',
        'visit_number',
        'hit_time',
        'hit_number',
        'hit_type',
        'hit_referer',
        'hit_page_path',
        'event_category',
        'event_action',
        'event_label',
        'device_screen_resolution',
        'utm_adcontent',
        'hit_date'

    ]
    df = df.copy()
    df = df.drop(columns=columns_to_drop, axis=1, errors='ignore')
    return df


def group_rare_data(df,
                    top_n_utm_keyword=30,
                    top_n_utm_source=50,
                    top_n_device_browser=10,
                    top_n_device_os=5,
                    top_n_utm_medium=25,
                    top_n_utm_campaign=100,
                    top_n_device_brand=50,
                    top_n_geo_city=100,
                    top_n_geo_country=50,
                    rare_category="Other"):
    df = df.copy()

    # group_rare_countries
    group_cols = {
        'utm_medium': top_n_utm_medium,
        'utm_source': top_n_utm_source,
        'utm_campaign': top_n_utm_campaign,
        'device_browser': top_n_device_browser,
        'device_brand': top_n_device_brand,
        'device_os': top_n_device_os,
        'geo_city': top_n_geo_city,
        'utm_keyword': top_n_utm_keyword,
        'geo_country': top_n_geo_country
    }

    for col, top_n in group_cols.items():
        top_values = df[col].value_counts().head(top_n).index
        df[col] = df[col].where(df[col].isin(top_values), rare_category)

    return df


def remove_duplicate_rows(df):
    df = df.copy()
    columns_to_check = [
        'utm_medium',
        'utm_source',
        'utm_campaign',
        'device_category',
        'device_os',
        'utm_keyword',
        'device_brand',
        'device_browser',
        'geo_country',
        'geo_city',
        'target_action'
    ]

    duplicates = df.duplicated(subset=columns_to_check, keep='first')
    df = df[~duplicates].copy()  # Добавляем .copy()
    return df


def main():
    df_s = pd.read_csv('data/ga_sessions.csv')
    df_s['session_id'] = df_s['session_id'].astype(str)
    df_h = pd.read_csv('data/ga_hits.csv')
    df_h['session_id'] = df_h['session_id'].astype(str)
    df = pd.merge(df_s, df_h, on='session_id', how='left')  # выгрузила и соединила два датасета
    df = df.dropna(subset=['event_action'])  # важно сразу удалить пустые строки

    target_actions = [  # это мои целевые действия по заданию
        'sub_car_claim_click',
        'sub_car_claim_submit_click',
        'sub_open_dialog_click',
        'sub_custom_question_submit_click',
        'sub_call_number_click',
        'sub_callback_submit_click',
        'sub_submit_success',
        'sub_car_request_submit_click'
    ]
    df['target_action'] = df['event_action'].isin(target_actions)

    df = kill_dublicate0(df)
    df = extract_and_update_utm(df)
    df = kill_NaN(df)
    df = kill_nan_device(df)
    df = filter_data(df)
    df = group_rare_data(df)
    df = remove_duplicate_rows(df)

    missing_values = ((df.isna().sum() / len(df)) * 100).sort_values()
    print(missing_values)
    columns_to_encode = [

        'utm_medium',
        'utm_source',
        'utm_campaign',
        'utm_keyword',
        'device_category',
        'device_brand',
        'device_os',
        'geo_country',
        'geo_city',
        'device_browser'

    ]

    for column in columns_to_encode:
        if column in df.columns:  # Проверяем, существует ли столбец в DataFrame
            num_unique = df[column].nunique()
            print(f"Столбец '{column}': {num_unique} уникальных значений")

    counts = df['target_action'].value_counts()
    print(counts)

    X = df.drop('target_action', axis=1)
    y = df['target_action'].astype(bool)

    categorical_features = X.select_dtypes(include='object').columns
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore'))

    ])
    preprocessor = ColumnTransformer(transformers=[
        ('categorical', categorical_transformer, categorical_features),
    ])

    rus = RandomUnderSampler(random_state=42, sampling_strategy=0.7) # добавляем даунсэмпинг

    models = {
        'LogisticRegression': LogisticRegression(solver='liblinear', max_iter=1000, penalty='l1', C=0.1),
        'RandomForestClassifier': RandomForestClassifier(max_depth=10, min_samples_leaf=5, max_features=0.7, n_estimators=500),
        'CatBoostClassifier': CatBoostClassifier(iterations=1000, learning_rate=0.01, depth=6, random_state=42, verbose=False),
        'MLPClassifier': MLPClassifier(random_state=42, max_iter=50, activation='logistic', hidden_layer_sizes=(256, 128), alpha=0.01, solver='adam',learning_rate_init=0.01, validation_fraction=0.2)
    }

    best_score = .0
    best_pipe = None
    best_model_name = None
    for model_name, model in models.items():
        pipe = make_pipeline(
            preprocessor,
            rus,
            model
        )
        score = cross_val_score(pipe, X, y, cv=10, scoring='accuracy')
        print(f'model: {type(model).__name__}, acc_mean: {score.mean():.4f}, acc_std: {score.std():.4f}')

        if score.mean() > best_score:
            best_score = score.mean()
            best_pipe = pipe
            best_model_name = model_name

    best_pipe.fit(X, y)
    print(f'Best model: {best_model_name}, ROC AUC: {best_score:.4f}')

    with open('Model_downsampling', 'wb') as file:
        dill.dump({
            'model': best_pipe,
            'metadata': {
                'name': 'Model downsampling',
                'author': 'Natasha',
                'version': 1,
                'Comm': 'Models without downsampling',
                'type': best_model_name,
                'accuracy': best_score
            }
        }, file)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Используем лучшую модель для прогноза
    y_pred = best_pipe.predict(X_test)
    y_pred_proba = best_pipe.predict_proba(X_test)[:, 1]

    # Рассчитаем метрики качества финальной модели
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    cm = confusion_matrix(y_test, y_pred)

    print("Confusion Matrix:")
    print(cm)
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-score: {f1:.4f}")
    print(f"ROC AUC: {roc_auc:.4f}")


if __name__ == '__main__':
    main()
