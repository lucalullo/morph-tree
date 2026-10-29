"""Platform wheel hook;C ABI avoids a dependency on the CPython extension ABI."""
import importlib.util
from pathlib import Path
from hatchling.builders.hooks.plugin.interface import BuildHookInterface

class CustomBuildHook(BuildHookInterface):
    def initialize(self,version,build_data):
        if self.target_name!='wheel':return
        path=Path(self.root)/'tools/build_native.py';spec=importlib.util.spec_from_file_location('morphtree_owned_build',path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        key=module.platform_key();native=module.NATIVE/'manifest.json'
        # Source installs compile locally. A wheel installation never executes
        # this hook. Existing owned binaries are accepted only by their hashes.
        import json
        valid=False
        if native.exists():
            v=json.loads(native.read_text('utf8'));p=module.NATIVE/v['library']
            valid=(v['platform']==key and p.is_file() and p.resolve().is_relative_to(module.NATIVE.resolve()) and module.sha(p)==v['binary_sha256'] and all(module.sha(module.NATIVE/n)==h for n,h in v['sources'].items()))
        if not valid:module.build()
        module.refresh_identity();build_data['pure_python']=False;build_data['tag']='py3-none-'+key
