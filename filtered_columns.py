def filter_columns(df, start_with=None, end_with=None, contains=None):
    filtered_cols = df.columns.tolist()

    # 시작 문자열 리스트로 필터링
    if start_with:
        if not isinstance(start_with, list):
            start_with = [start_with]
        filtered_cols = [col for col in filtered_cols if any(col.startswith(start) for start in start_with)]

    # 끝 문자열 리스트로 필터링
    if end_with:
        if not isinstance(end_with, list):
            end_with = [end_with]
        filtered_cols = [col for col in filtered_cols if any(col.endswith(end) for end in end_with)]

    # 특정 문자열 포함 필터링
    if contains:
        if not isinstance(contains, list):
            contains = [contains]
        filtered_cols = [col for col in filtered_cols if any(contain in col for contain in contains)]

    return df[filtered_cols]
