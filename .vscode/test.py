import json
filename = "C:/Users/CL1Student28/test/.vscode/intents.txt"

dict1 = {}

with open (filename) as fh:
    for line in fh:

        command, description = line.strip().split(None, 1)

        dict1[command] = description.strip()

out_file = open("intents.json", "w")
json.dump(dict1, out_file, indent = 4, sort_keys = False)
out_file.close()
 