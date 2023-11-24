import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score


def merge_data(raw, labeling, col_rename=0):

    '''
    raw 데이터 + labeling 데이터
    기준 컬럼 : email

    '''
    if col_rename == 1:
        raw = raw.rename(columns={'SAMPLE_EMAIL':'EMAIL'})
    renamed_labeling=labeling.rename(columns={'SAMPLE_EMAIL':'EMAIL'})
    merged=pd.merge(raw, renamed_labeling, on='EMAIL')

    result = merged.rename(columns={'DIAG_NM':'target'})

    return result


def split_train_test(date_df, percentage = 0.8):
    train_num = int(len(date_df)*percentage)

    x_df = date_df.drop(['EMAIL', 'target'], axis=1)

    #target 값 one-hot encoding
    y_df = pd.get_dummies(date_df['target'])

    train_X = x_df[:train_num].values
    train_y = y_df[:train_num].values
    test_X = x_df[train_num:].values
    test_y = y_df[train_num:].values

    return train_X, train_y, test_X, test_y


def metrics(test_y, y_predict):
    f1 = round(f1_score(test_y, y_predict, average='micro'), ndigits= 5)
    precision = round(precision_score(test_y, y_predict, average='macro'), ndigits=5)
    recall = round(recall_score(test_y, y_predict, average='macro'), ndigits=5)
    print(f'fl:{f1}, 정밀도 : {precision}, 재현율:{recall}')
    return f1, precision, recall