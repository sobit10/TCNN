import numpy as np


def stoi(s):
    try:
        return int(s)
    except ValueError:
        print("Error: Input string is not a valid integer.")
        return None


feature_strings = np.load('pre_evaluated/data.npy', allow_pickle=True)
target_strings = np.load('pre_evaluated/target.npy', allow_pickle=True)

features = [stoi(s) for s in feature_strings]

targets = [stoi(s) for s in target_strings]


def pseq(feature_sequences, target_sequences):
    paired_sequences = []
    for features, targets in zip(feature_sequences, target_sequences):
        if len(features) != len(targets):
            print("Error: Length of feature sequence does not match length of target sequence.")
            continue
        paired_sequences.append((features, targets))
    return paired_sequences


# Example sequences of features and targets
feature_sequences = np.load('pre_evaluated/data.npy', allow_pickle=True)
target_sequences = np.load('pre_evaluated/target.npy', allow_pickle=True)

paired_sequences = pseq(feature_sequences, target_sequences)

for idx, pair in enumerate(paired_sequences):
    print(f"Pair {idx + 1}:")
    print("Features:", pair[0])
    print("Targets:", pair[1])
