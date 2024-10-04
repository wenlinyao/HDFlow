import json
import copy

if __name__ == "__main__":
    # load json file
    with open("MathInstruct.json", "r") as f:
        data = json.load(f)
    new_instances = []
    existing_str = set()

    print("len(data):", len(data))

    for i, instance in enumerate(data):
        prompt_str = instance["instruction"]
        if "PoT" in instance["source"]:
            if " Let's " in prompt_str:
                fields = prompt_str.split(" Let's ")[:-1]
                prompt_str = " Let's ".join(fields)
            elif " Please " in prompt_str:
                fields = prompt_str.split(" Please ")[:-1]
                prompt_str = " Please ".join(fields)
            else:
                print(prompt_str)
                input("continue?")
        
        if prompt_str in existing_str:
            continue
        existing_str.add(prompt_str)

        new_instance = {}
        new_instance["Id"] = f"{i}__" + "-".join(instance["source"].split("/")[1:]).replace(".json", "")
        new_instance["new_instruction"] = prompt_str

        new_instances.append(new_instance)
    
    print("len(new_instances):", len(new_instances))

    # split every 50000 instances into a file

    n = len(new_instances)
    print("n:", n)
    part_size = 50000
    print("part_size:", part_size)

    for i in range(n // part_size + 1):
        start = i * part_size
        end = (i+1) * part_size
        if i == n // part_size:
            end = n
        print("start:", start, "end:", end)
        part_data = new_instances[start:end]
        print("len(part_data):", len(part_data))
        json.dump(part_data, open(f"MathInstruct_clean_part{i}.json", "w"), indent=2)



