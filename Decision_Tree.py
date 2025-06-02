import csv
from math import log2
from collections import defaultdict

def read_data(filepath):
    with open(filepath, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)
        data = [row for row in reader]
    return header, data

def print_data(data):
    for row in data:
        print(row)
    return data
    
def result_counts(rows):
    counts = {}
    for row in rows:
        label = row[-1]
        counts[label] = counts.get(label, 0) + 1
    return counts

def split_by_feature(rows, column):
    splits = defaultdict(list)
    for row in rows:
        splits[row[column]].append(row)
    return splits

def entropy(rows): 
    counts = result_counts(rows)
    length = len(rows)
    probabilities = [count / length for count in counts.values()]
    if length == 0:
        return 0
    return -sum(p * log2(p) for p in probabilities if p > 0)

def information_gain(left, right, current_uncertainty):
    p = float(len(left)) / (len(left) + len(right))
    return current_uncertainty - p * entropy(left) - (1 - p) * entropy(right)

def find_best_split(rows, header):
    header = header
    best_gain = 0
    best_feature = None
    best_splits = None
    current_uncertainty = entropy(rows)
    feature_number = len(rows[0]) - 1

    print(f"Total entropy of current node: {current_uncertainty:.3f}\n")

    for col in range(feature_number):
        print(f"Evaluating attribute: {header[col]}")
        splits = split_by_feature(rows, col)
        if len(splits) <= 1:
            print(" Only one unique value. Skipping.\n")
            continue

        weighted_entropy = 0.0
        for value, subset in splits.items():
            count_dict = result_counts(subset)
            entropy_val = entropy(subset)
            weighted = len(subset) / len(rows)
            weighted_entropy += weighted * entropy_val

            count_str = ", ".join([f"{count} {label}" for label, count in count_dict.items()])
            print(f"  {value} ({len(subset)}): {count_str} → Entropy = {entropy_val:.3f}")

        gain = current_uncertainty - weighted_entropy
        print(f"  Gain(S, {header[col]}) = {current_uncertainty:.3f} - Weighted Entropy = {weighted_entropy:.3f} → Gain = {gain:.3f}\n")

        if gain > best_gain:
            best_gain = gain
            best_feature = col
            best_splits = splits

    if best_feature is not None:
        print(f"Best attribute : {header[best_feature]} (Gain = {best_gain:.3f})\n")
    else:
        print("No good split found.\n")

    return best_gain, best_feature, best_splits


def build_tree(rows, header):
    if len(rows) == 0:
        return None
    if len(set(row[-1] for row in rows)) == 1:
        return rows[0][-1]

    gain, feature, splits = find_best_split(rows,header)
    if gain == 0:
        return result_counts(rows)

    tree = {feature: {}}
    for value, subset in splits.items():
        tree[feature][value] = build_tree(subset, header)

    return tree

def print_tree(tree, header, depth=0):
    if isinstance(tree, dict):
        if len(tree) == 1 and all(isinstance(k, int) for k in tree.keys()): 
            feature = next(iter(tree))  # int index
            print("  " * depth + f"{header[feature]}:")
            branches = tree[feature]
            for value, subtree in branches.items():
                print("  " * (depth + 1) + f"{value} ->")
                print_tree(subtree, header, depth + 2)
        else:
            total = sum(tree.values())
            for label, count in tree.items():
                percentage = 100 * count / total
                print("  " * depth + f"Prediction: {label} ({count} samples, %{percentage:.2f})")
    else:
        print("  " * depth + f"Prediction: {tree}")



def classify(row, tree):
    while isinstance(tree, dict):
        feature_index = next(iter(tree))
        feature_index = next(iter(tree))
        try:
            feature_index = int(feature_index)
        except:
            break
        next_tree = tree[feature_index]

        feature_value = row[feature_index]

        if feature_value in next_tree:
            tree = next_tree[feature_value]
        else: 
            return "Unknown"

    if isinstance(tree, str):
        return tree
    elif isinstance(tree, dict):
        total = sum(tree.values())
        return {label: round(100 * count / total, 2) for label, count in tree.items()}
    else:
        return "Unknown"

def main ():
    filepath = 'weather.csv'  
    header, data = read_data(filepath)

    tree = build_tree(data,header)
    print("Decision Tree:")
    print_tree(tree, header)
    inputs = []
    for feature in header[:-1]: 
        value = input(f"Enter value for '{feature}': ").strip()
        inputs.append(value)
    prediction = classify(inputs, tree)
    print("\nPrediction:", prediction)
    
    '''
    #testing read_data function
    print("Testing read_data function")
    print("Header:", header)
    print("Data:")
    print_data(data)
    '''
    '''
    #testing result_counts function
    print("\nTesting result_counts function")
    counts = result_counts(data)
    print("Counts:", counts)
    '''

    '''
    #testing entropy function
    print("\nTesting entropy function")
    current_uncertainty = entropy(data)
    print("Current uncertainty (entropy):", current_uncertainty)
    '''

    '''
    #testing information_gain function
    print("\nTesting information gain ")
    left = [row for row in data if row[0] == 'sunny']
    right = [row for row in data if row[0] != 'sunny']
    print("Left split (sunny):", left)
    print("Right split (not sunny):", right)
    gain = information_gain(left, right, current_uncertainty)
    print("Information Gain:", gain) 
    '''

    '''
    #testing split_by_feature function
    print("\nTesting split_by_feature function")
    splits = split_by_feature(data, 0)
    print("Splits:", splits)
    '''

    '''
    #testing find_best_split function
    print("\nTesting find_best_split function")
    best_gain, best_feature, best_splits = find_best_split(data, header)
    print("Best gain:", best_gain)
    print("Best feature:", best_feature)
    print("Best splits:", best_splits)
    '''

    '''
    example_row = data[223] # For breast_cancer dataset
    prediction = classify(example_row, tree)
    print("Prediction:", prediction)
    '''
    

if __name__ == "__main__":
    main()
    