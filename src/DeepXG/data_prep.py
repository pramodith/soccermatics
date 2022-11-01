import numpy as np
from mplsoccer import Sbopen
from src.constants import *
from collections import defaultdict
from utils import get_shot_angle, get_shot_distance, get_pass_shot_zones
import json
from ast import literal_eval

TOURNAMENT_ID = 11
TRAIN_SEASON_IDS = [37, 38, 39, 40, 41, 21, 22, 23, 24, 26, 27, 2, 1, 4, 42]
VALIDATION_SEASON_IDS = [25]
TEST_SEASON_IDS = [90]


def get_laliga_match_ids(season_ids, output_dir="../../data/deepXG", mode=TRAIN):
    parser = Sbopen()
    la_liga_match_ids = []
    for season in season_ids:
        match_df = parser.match(competition_id=TOURNAMENT_ID, season_id=season)
        la_liga_match_ids.extend(match_df[MATCH_ID].values)

    df = pd.DataFrame(columns=[MATCH_ID], data=la_liga_match_ids)
    df.to_csv(f"{output_dir}/match_ids_{mode}.csv", index=False)


def create_dataframe(mode=TRAIN):
    match_df = pd.read_csv(f"../../data/deepXG/match_ids_{mode}.csv")
    parser = Sbopen()
    data_contents = defaultdict(list)
    for _, row in match_df.iterrows():
        event_data, _, _, _ = parser.event(row[MATCH_ID])
        event_data.sort_values(by=[INDEX], inplace=True)
        prev_possession_team = None
        pass_sequence = []
        for _, event_row in event_data.iterrows():

            if event_row[TYPE_NAME] == PASS and event_row[POSSESSION_TEAM_NAME] == prev_possession_team:
                pass_sequence.append((event_row[X], event_row[Y]))
                if not pd.isna(event_row[PASS_SHOT_ASSIST]):
                    data_contents[PASS_TECHNIQUE_NAME].append(event_row[TECHNIQUE_NAME])

            elif event_row[TYPE_NAME] == SHOT:
                data_contents[SHOT_LOCATION_X].append(event_row[X])
                data_contents[SHOT_LOCATION_Y].append(event_row[Y])
                data_contents[SHOT_DISTANCE].append(get_shot_distance(event_row[X], event_row[Y]))
                data_contents[SHOT_ANGLE].append(get_shot_angle(event_row[X], event_row[Y]))
                data_contents[TECHNIQUE_NAME].append(event_row[TECHNIQUE_NAME])
                data_contents[SUB_TYPE_NAME].append(event_row[SUB_TYPE_NAME])
                data_contents[UNDER_PRESSURE].append(event_row[UNDER_PRESSURE])
                data_contents[BODY_PART_NAME].append(event_row[BODY_PART_NAME])
                data_contents[OUTCOME_NAME].append(event_row[OUTCOME_NAME])
                data_contents[POSITION_NAME].append(event_row[POSITION_NAME])
                if len(data_contents[PASS_TECHNIQUE_NAME]) != len(data_contents[POSITION_NAME]):
                    data_contents[PASS_TECHNIQUE_NAME].append(np.NAN)
                if event_row[SUB_TYPE_NAME] in [FREE_KICK, PENALTY]:
                    pass_sequence = []
                data_contents[PASS_SEQUENCE].append(pass_sequence)
                pass_sequence = []

            elif not event_row[TYPE_NAME] in [PASS, CARRY, DRIBBLE, BALL_RECEIPT, DRIBBLED_PAST, PRESSURE, DUEL, MISCONTROL] or \
                    event_row[POSSESSION_TEAM_NAME] != prev_possession_team:
                if event_row[POSSESSION_TEAM_NAME] != prev_possession_team:
                    prev_possession_team = event_row[POSSESSION_TEAM_NAME]
                pass_sequence = []

    pd.DataFrame.from_dict(data_contents).to_csv(f"../../data/deepXG/{mode}.csv",index=False)


def data_transformation(df: pd.DataFrame, mode=TRAIN):
    # Get the passing and shot zones
    df[PASS_SEQUENCE] = df[PASS_SEQUENCE].apply(literal_eval)
    pass_zones = []
    for _, row in df.iterrows():
        passes = list(zip(*row[PASS_SEQUENCE]))
        try:
            pass_zones.append([ZONE2ID[p] for p in get_pass_shot_zones(passes[0], passes[1])])
        except Exception as e:
            pass_zones.append([80])

    df[PASS_ZONES] = pass_zones


    df[SHOT_ZONES] = get_pass_shot_zones(df[SHOT_LOCATION_X].values, df[SHOT_LOCATION_Y].values)
    df[SHOT_ZONES] = df[SHOT_ZONES].apply(lambda x: ZONE2ID[x])

    if mode == TRAIN:
        # Convert Categorical variables to ids
        cat2id = {}
        for col in CATEGORICAL_VARIABLES:
            df[col] = df[col].astype("category")
            cat2id[col] = df[col].cat.categories.tolist()
            df[col] = df[col].cat.codes

        # Standardize continuous variables
        cont2stdnorm = {}
        for col in CONTINUOUS_VARIABLES:
            mean = df[col].mean()
            std = df[col].std()
            cont2stdnorm[col] = {"mean": mean, "std": std}
            df[col] = (df[col] - mean) / std

        # Save dict mapping category to ids
        with open("../../data/deepXG/cat2id.json", 'w') as f, open("../../data/deepXG/cont2stdnorm.json", 'w') as g:
            json.dump(cat2id, f)
            json.dump(cont2stdnorm, g)

    else:
        with open("../../data/deepXG/cat2id.json", 'r') as f, open("../../data/deepXG/cont2stdnorm.json", 'r') as g:
            cat2id = json.load(f)
            cont2stdnorm = json.load(g)

        for col in CATEGORICAL_VARIABLES:
            print(col, cat2id[col])
            df[col] = df[col].apply(lambda x: cat2id[col].index(x) if x in cat2id[col] else len(cat2id[col]))

        for col in CONTINUOUS_VARIABLES:
            df[col] = (df[col] - cont2stdnorm[col]["mean"]) / cont2stdnorm[col]["std"]

    df.to_csv(f"../../data/deepXG/{mode}.csv",index=False)
    return df


if __name__ == "__main__":
    get_laliga_match_ids(SEASON_IDS, TRAIN)
    get_laliga_match_ids(VALIDATION_SET, VAL)
    get_laliga_match_ids(TEST_SET, TEST)
    create_dataframe(TRAIN)
    create_dataframe(VAL)
    create_dataframe(TEST)
    train_df = pd.read_csv("../../data/deepXG/train.csv")
    val_df = pd.read_csv("../../data/deepXG/val.csv")
    test_df = pd.read_csv("../../data/deepXG/test.csv")
    data_transformation(train_df, TRAIN)
    data_transformation(val_df, VAL)
    data_transformation(test_df, TEST)