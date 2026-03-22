import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

splits = ["train", "val", "test"]

for split in splits:
    labels_folder = os.path.join(BASE_DIR, split, "labels")

    if not os.path.exists(labels_folder):
        print(f"Skipping {split}, labels folder not found")
        continue

    print(f"Processing {split}...")

    for filename in os.listdir(labels_folder):
        if filename.endswith(".txt"):
            file_path = os.path.join(labels_folder, filename)

            with open(file_path, "r") as f:
                lines = f.readlines()

            new_lines = []
            for line in lines:
                parts = line.strip().split()

                if len(parts) > 0:
                    class_id = int(parts[0])
                    new_class_id = class_id + 1  # shift 0→1, 1→2, etc.
                    parts[0] = str(new_class_id)

                new_lines.append(" ".join(parts))

            with open(file_path, "w") as f:
                f.write("\n".join(new_lines))

    print(f"{split} done.")

print("All splits processed.")