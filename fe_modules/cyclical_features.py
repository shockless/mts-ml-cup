def generate_cyclical_features(df, col_name, period, start_num=0):
    kwargs = {
        f'sin_{col_name}' : lambda x: np.sin(2 * np.pi * (df[col_name]-start_num) / period),
        f'cos_{col_name}' : lambda x: np.cos(2 * np.pi * (df[col_name]-start_num) / period)
             }
    return df.assign(**kwargs)
