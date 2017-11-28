import os
import sys
import subprocess
import os.path

ENV_VAR = 'BIZHAWK'

bizhawk_path = os.environ.get(ENV_VAR)
if bizhawk_path is None:
    print("no bizhawk path found in environment variable: {}".format(ENV_VAR))
    sys.exit()

print("Attempting to start bizhawk from: {}".format(bizhawk_path))

# it just looks for 1 backslash first is escape char
if not bizhawk_path.endswith("\\"):
    bizhawk_path = bizhawk_path + "\\"

if not os.path.isdir(bizhawk_path + "isos"):
    print("please create {}isos directory and put in \
           it the mario iso named: mario.z64".format(bizhawk_path))
    sys.exit()
mario_path = bizhawk_path + "isos\\mario.z64"
if not os.path.isfile(mario_path):
    print("add {}".format(mario_path))
    sys.exit()

cmd = "{}EmuHawk.exe".format(bizhawk_path) + " {}".format(mario_path)
print("run command: {}".format(cmd))
subprocess.run(["{}EmuHawk.exe".format(bizhawk_path), "{}".format(mario_path)])
