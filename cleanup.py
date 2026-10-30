import os, re, sys

FOLDERS = ["uploads", "processed"]   # folders to clean
START = 41                           # delete datasets numbered 41 and above
PROTECTED = 35                       # trained dataset: never touched

DELETE = "--delete" in sys.argv
pattern = re.compile(r"telco dataset-(\d+)\.csv", re.IGNORECASE)


def should_delete(name):
    if name.lower().startswith("active_model"):
        return False
    m = pattern.search(name)
    if not m:
        return False
    num = int(m.group(1))
    return num >= START and num != PROTECTED


found = []
for folder in FOLDERS:
    if not os.path.isdir(folder):
        print(f"(skipping missing folder: {folder})")
        continue
    for name in sorted(os.listdir(folder)):
        path = os.path.join(folder, name)
        if os.path.isfile(path) and should_delete(name):
            found.append(path)

print(f"\n{'DELETING' if DELETE else 'WOULD DELETE'} ({len(found)} files, datasets {START} and above):")
for p in found:
    print("  ", p)

if DELETE:
    for p in found:
        os.remove(p)
    print("\nDone.")
else:
    print("\nDry run only. Run again with --delete to remove these.")