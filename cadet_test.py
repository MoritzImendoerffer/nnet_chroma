import shutil
import os
import platform
from pathlib import Path
from cadet import Cadet

# Either ensure CADET is on your PATH (e.g. by installing via conda)
# OR
# provide the path to the CADET installation
# E.g.
#     windows: C:\Users\<username>\cadet
#     linux: ~/cadet/bin
# would be set by:

install_path = None

executable = 'cadet-cli'
if install_path is None:
    try:
        if platform.system() == 'Windows':
            executable += '.exe'
        executable_path = Path(shutil.which(executable))
    except TypeError:
        raise FileNotFoundError(
            "CADET could not be found. Please set an install path"
        )
    install_path = executable_path.parent.parent

install_path = Path(install_path)
cadet_bin_path = install_path / "bin" / executable

if cadet_bin_path.exists():
    Cadet.cadet_path = cadet_bin_path
else:
    raise FileNotFoundError(
        "CADET could not be found. Please check the path"
    )

cadet_lib_path = install_path / "lib"
try:
    if cadet_lib_path.as_posix() not in os.environ['LD_LIBRARY_PATH']:
        os.environ['LD_LIBRARY_PATH'] = \
            cadet_lib_path.as_posix() \
            + os.pathsep \
            + os.environ['LD_LIBRARY_PATH']
except KeyError:
    os.environ['LD_LIBRARY_PATH'] = cadet_lib_path.as_posix()

lwe_executable = 'createLWE'
if platform.system() == 'Windows':
    lwe_executable += '.exe'
lwe_path = install_path / "bin" / lwe_executable