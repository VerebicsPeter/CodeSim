import os
import configparser
import pandas as pd


config = configparser.ConfigParser()
config.read('paths.ini')
DATA_PATH = config['DEFAULT']['DataPath']
META_PATH = config['DEFAULT']['MetaPath']


def python_solutions(pid: str) -> list[str]:
    """ Returns the paths of Python solutions for a given problem (`pid`). """
    path = os.path.abspath(os.path.join(DATA_PATH, pid, 'Python'))
    if not os.path.isdir(path):
        return []
    return os.listdir(path)  # return the paths of the solutions


def init_metadata_df(pid: str) -> pd.DataFrame | None:
    """ Initializes and returns a dataframe containing the metadata of a problem (`pid`). """
    path = os.path.abspath(os.path.join(META_PATH, f'{pid}.csv'))
    if not os.path.isfile(path):
        return None
    df: pd.DataFrame = pd.read_csv(path)
    # Filter the data:
    df = df[['submission_id','status','language','user_id','date','accuracy']]
    df = df.loc['Python'  == df['language']]
    df = df.loc[1577836800 < df['date']]  # > 2020 jan 1
    df = df[['submission_id','status']]
    return df


def read_solution_data(pid: str, sid: str, df: pd.DataFrame | None = None) -> pd.DataFrame:
    """ Reads the data of a solution identified by the problem and solution IDs (`pid`, `sid`). """
    if df is None: df = init_metadata_df(pid)
    return df.loc[(df['submission_id'] == sid)]


def read_solution_file(pid: str, sid: str) -> str:
    """ Reads the source file of a solution identified by the problem and solution IDs (`pid`, `sid`). """
    path = os.path.abspath(os.path.join(DATA_PATH, pid, 'Python'))
    fp = os.path.abspath(os.path.join(path, f'{sid}.py'))
    with open(fp, 'r', encoding='utf-8') as f:
        return f.read()
