import pandas as pd

def handling_NaN(nrows = 420003):
    file_path = 'Data_Fitness/athletes.csv'
    df_sample = pd.read_csv(file_path, nrows=nrows)
    columns_to_drop = ['athlete_id', 'name', 'retrieved_datetime', 'team', 'affiliate', 'fran', 'helen', 'grace', 'filthy50', 'fgonebad', 'run400', 'run5k', 'pullups', 'eat', 'train', 'background', 'experience', 'schedule', 'howlong']
    columns_to_drop = [col for col in columns_to_drop if col in df_sample.columns]
    df = df_sample.drop(columns=columns_to_drop)
    df.dropna(thresh=0.85, inplace=True)

    df.to_csv("Assignment_2_Fx/Assignment_2_Data_sets/selected_data.csv", index=False)

    return df
