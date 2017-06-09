
import configparser
import pickle
from data import load_data

config = configparser.ConfigParser()
config.read_file(open('default.cfg', "r"))

if __name__ == "__main__":
    TEXT_PATH = config['DEFAULT'].get("TEXT_PATH")
    train_data = load_data(TEXT_PATH)
    pickle.dump(train_data, open("data/train_data.pyObj", "wb"))    

