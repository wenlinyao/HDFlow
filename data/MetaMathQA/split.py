import json
import random

if __name__ == "__main__":
    random.seed(11)
    data = json.load(open("MetaMathQA-395K.json"))

    random.shuffle(data)

    # split into 20 parts

    n = len(data)
    print("n:", n)
    part_size = n // 20
    print("part_size:", part_size)

    for i in range(20):
        start = i * part_size
        end = (i+1) * part_size
        if i == 19:
            end = n
        print("start:", start, "end:", end)
        part_data = data[start:end]
        json.dump(part_data, open(f"MetaMathQA-395K_{i}.json", "w"), indent=2)