import os
import sys
import subprocess
import os.path

ENV_VAR_BIZ = 'BIZHAWK'
ENV_VAR_MKDL_LUA = 'MKDL_LUA'


def start_mario(num_env=-1):
    bizhawk_path = os.environ.get(ENV_VAR_BIZ)
    if bizhawk_path is None:
        print("no bizhawk path found in environment variable: {}".format(ENV_VAR_BIZ))
        sys.exit()

    mkdl_lua_path = os.environ.get(ENV_VAR_MKDL_LUA)
    if mkdl_lua_path is None:
        print("show us the mkdl lua path in the environment variable: {}".format(ENV_VAR_MKDL_LUA))
        sys.exit()

    print("Attempting to start bizhawk from: {}".format(bizhawk_path))
    print("do not forget to save a slot 1 if you start mario via this script")
    # it just looks for 1 backslash first is escape char
    if not bizhawk_path.endswith("\\"):
        bizhawk_path = bizhawk_path + "\\"

    if not mkdl_lua_path.endswith("\\"):
        mkdl_lua_path = mkdl_lua_path + "\\"

    if num_env is -1:
        lua_file_path = '{}new_mario_env.lua'.format(mkdl_lua_path)
    else:
        lua_file_path = '{}new_mario_env{}.lua'.format(mkdl_lua_path, num_env)

    if not os.path.isdir(bizhawk_path + "isos"):
        print("please create {}isos directory and put in \
               it the mario iso named: mario.z64".format(bizhawk_path))
        sys.exit()
    mario_path = bizhawk_path + "isos\\mario.z64"
    if not os.path.isfile(mario_path):
        print("add {}".format(mario_path))
        sys.exit()

    proc = subprocess.Popen(["{}EmuHawk.exe".format(bizhawk_path),
                             "--load-slot=1",
                             "--lua={}".format(lua_file_path),
                             "{}".format(mario_path)])
    print('running: {}'.format(proc.args))
    return proc
