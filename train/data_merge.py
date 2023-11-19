import pandas as pd

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