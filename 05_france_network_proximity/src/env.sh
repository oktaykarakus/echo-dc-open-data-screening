# source from 05_france_network_proximity/ ; forces all temp/cache into the phase folder
export F2B="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)"
export TMPDIR="$F2B/.tmp" TMP="$F2B/.tmp" TEMP="$F2B/.tmp" XDG_CACHE_HOME="$F2B/.tmp" MPLCONFIGDIR="$F2B/.tmp" PYTHONPYCACHEPREFIX="$F2B/.tmp/pyc" PYTHONDONTWRITEBYTECODE=1 CPL_TMPDIR="$F2B/.tmp" GDAL_CACHEMAX=64 PROJ_USER_WRITABLE_DIRECTORY="$F2B/.tmp"
