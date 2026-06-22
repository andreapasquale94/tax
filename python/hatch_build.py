"""Hatchling build hook: vendor the tax headers into the wheel before file collection."""
from hatchling.builders.hooks.plugin.interface import BuildHookInterface

class CustomBuildHook(BuildHookInterface):
    def initialize(self, version, build_data):
        import sys, pathlib
        sys.path.insert(0, str(pathlib.Path(self.root)))   # ensure `tax` importable
        from tax._vendor import sync_from_repo
        sync_from_repo()
