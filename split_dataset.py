import os
import shutil
import errno
import sys
import random

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Expected dataset root path")
        exit(1)
    for root, dirs, files in os.walk(sys.argv[1]):
        for file in files:
            if file.endswith(".jpg") and random.random() < 0.2:
                try:
                    shutil.move(root + "/" + file, root.replace("training", "testing") + "/" + file)
                except IOError as e:
                    if e.errno != errno.ENOENT:
                        raise
                    os.makedirs(os.path.dirname(root.replace("training", "testing") + "/" + file))
                    shutil.move(root + "/" + file, root.replace("training", "testing") + "/" + file)
