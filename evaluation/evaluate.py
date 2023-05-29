import numpy as np
import os


def load_data_numpy(path):
    eval_file = np.load(path)
    fitness = eval_file["fitness"]
    rms = eval_file["rms"]
    return fitness, rms


if __name__ == "__main__":
    base_dir = os.getcwd()
    base = os.path.join(base_dir, "evaluation", "data_init")
    base = os.path.join(base_dir, "evaluation", "data_vxl")
    base = os.path.join(base_dir, "evaluation", "data_dist")
    for folder_name in os.listdir(base):
        folder_path = os.path.join(base, folder_name)
        fitness, rms = load_data_numpy(folder_path)
        print(folder_name)
        print(fitness.shape)
        print(fitness)
        print(rms)
        print("--" * 50)
