import re

LABEL_MAP = {
    (1, "Normal"): 0, (1, "Abnormal"): 1,
    (2, "Normal"): 2, (2, "Abnormal"): 3,
    (3, "Normal"): 4, (3, "Abnormal"): 5,
}

LABEL_NAMES = {
    0: "Machine 1 - Normal",   1: "Machine 1 - Abnormal",
    2: "Machine 2 - Normal",   3: "Machine 2 - Abnormal",
    4: "Machine 3 - Normal",   5: "Machine 3 - Abnormal",
}

def get_label(machine_folder, status):
    match = re.search(r'\d+', machine_folder)
    if not match:
        raise ValueError(f"Cannot parse machine number from: {machine_folder}")
    key = (int(match.group()), status)
    if key not in LABEL_MAP:
        raise ValueError(f"Unknown combination: {key}")
    return LABEL_MAP[key]
