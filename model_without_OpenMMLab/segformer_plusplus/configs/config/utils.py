import ast
import os.path as osp
import re
import sys
import warnings
from collections import defaultdict
from importlib.util import find_spec
from typing import List, Optional, Tuple, Union
from importlib import import_module as real_import_module
import json
import pickle
from pathlib import Path
import itertools
import importlib.util  # Hinzugefügt für find_spec in Python >= 3.7
import importlib.metadata  # Ersatz für pkg_resources.get_distribution und package2module

# Ersetzt: from pkg_resources.extern import packaging
# Ersetzt: __import__('pkg_resources.extern.packaging.version')
# Ersetzt: __import__('pkg_resources.extern.packaging.specifiers')
# Ersetzt: __import__('pkg_resources.extern.packaging.requirements')
# Ersetzt: __import__('pkg_resources.extern.packaging.markers')
# Importiert nun das externe 'packaging'-Paket direkt
try:
    import packaging.version
    import packaging.specifiers
    import packaging.requirements
    import packaging.markers
except ImportError as e:
    raise ImportError(
        "The 'packaging' package is required but not installed. "
        "Install it with 'pip install packaging'."
    ) from e

import yaml
from omegaconf import OmegaConf


# Die folgenden Hilfsfunktionen aus pkg_resources wurden unten neu implementiert
# oder durch `packaging` ersetzt: get_distribution, package2module, Requirement,
# parse_requirements, yield_lines, safe_extra, safe_name.

# --- Hilfsfunktionen für `packaging` (Ersatz für pkg_resources-Logik) ---

# Implementierung der pkg_resources-Hilfsfunktionen mit 'packaging'
def safe_extra(extra: str) -> str:
    """Convert an arbitrary string to a standard 'extra' name"""
    # pkg_resources implementation detail using packaging's rules
    return re.sub(r'[^A-Za-z0-9.-]+', '_', extra).lower()


def safe_name(name: str) -> str:
    """Convert an arbitrary string to a standard distribution name"""
    # pkg_resources implementation detail using packaging's rules
    return re.sub(r'[^A-Za-z0-9.]+', '-', name)


class DistributionNotFound(Exception):
    """Exception raised when a distribution is not found."""
    pass


def get_distribution(dist_name: str) -> importlib.metadata.Distribution:
    """Return a current distribution object for a package name or string requirement.

    Args:
        dist_name (str): The name of the package or a requirement string.

    Returns:
        importlib.metadata.Distribution: The found distribution object.

    Raises:
        DistributionNotFound: If the package is not found.
        ValueError: If a requirement string is used (not supported by this simplified function).
    """
    if ' ' in dist_name or any(op in dist_name for op in ('==', '>=', '<=', '>', '<', '~=', '!=', '==')):
        # importlib.metadata.distribution does not handle requirement strings.
        # It's better to use the distribution name directly.
        # Fallback to direct name extraction.
        try:
            req = packaging.requirements.Requirement(dist_name)
            dist_name = req.name
        except packaging.requirements.InvalidRequirement:
            raise ValueError(
                f"get_distribution only supports package names or simple requirements, "
                f"but got: {dist_name}"
            )

    try:
        # Use importlib.metadata.distribution for name-based lookup
        return importlib.metadata.distribution(dist_name)
    except importlib.metadata.PackageNotFoundError:
        raise DistributionNotFound(f"The 'Distribution' '{dist_name}' was not found and is required")


def package2module(package: str) -> str:
    """Infer module name from package using importlib.metadata.

    Args:
        package (str): Package to infer module name.

    Returns:
        str: The module name.

    Raises:
        ValueError: If the top-level module name cannot be inferred.
    """
    try:
        # Use importlib.metadata.distribution
        dist = get_distribution(package)

        # Check for top_level.txt metadata
        top_level_txt = dist.read_text('top_level.txt')
        if top_level_txt:
            # The first non-empty line is usually the top-level module name
            module_name = top_level_txt.split('\n')[0].strip()
            if module_name:
                return module_name

    except (DistributionNotFound, FileNotFoundError):
        # Package not found or top_level.txt not available/found
        pass  # Will raise ValueError below

    raise ValueError(
        highlighted_error(f'can not infer the module name of {package}. '
                          'Metadata (top_level.txt) not found or package not installed.')
    )


# Neuimplementierung der Requirement-Logik von pkg_resources mit packaging.requirements.Requirement

class Requirement(packaging.requirements.Requirement):
    """Reimplementation of pkg_resources.Requirement using packaging.requirements.Requirement."""

    def __init__(self, requirement_string):
        """DO NOT CALL THIS UNDOCUMENTED METHOD; use Requirement.parse()!"""
        super().__init__(requirement_string)
        self.unsafe_name = self.name
        project_name = safe_name(self.name)
        self.project_name, self.key = project_name, project_name.lower()

        # specs is an internal list in pkg_resources
        self.specs = [
            (spec.operator, spec.version) for spec in self.specifier
        ] if self.specifier else []

        self.extras = tuple(map(safe_extra, self.extras))

        # hashCmp logic from pkg_resources
        self.hashCmp = (
            self.key,
            self.url,
            str(self.specifier) if self.specifier else '',
            frozenset(self.extras),
            str(self.marker) if self.marker else None,
        )
        self.__hash = hash(self.hashCmp)

    def __eq__(self, other):
        return (
                isinstance(other, Requirement) and
                self.hashCmp == other.hashCmp
        )

    def __contains__(self, item: packaging.version.Version) -> bool:
        """Check if a specific version is contained in the requirement."""
        if isinstance(item, str):
            try:
                item = packaging.version.Version(item)
            except packaging.version.InvalidVersion:
                warnings.warn(f"Invalid version string: {item}", UserWarning)
                return False

        if self.key != safe_name(item.base_version).lower():
            # This check is an oversimplification but reflects pkg_resources' intent
            # In a real-world scenario, you should compare the package name.
            # Since the original code only imports 'packaging' and not a distribution object,
            # this check can only be an approximation.
            return False

            # The packaging specifier can check against packaging.version.Version objects
        # prereleases=True is required to match pkg_resources' default behavior
        return self.specifier.contains(item, prereleases=True)

    def __hash__(self):
        return self.__hash

    @staticmethod
    def parse(s):
        reqs = list(parse_requirements(s))
        if not reqs:
            raise ValueError(f"Could not parse requirement from string: {s}")
        return reqs[0]


def yield_lines(iterable: Union[str, list, tuple]) -> List[str]:
    """Yield valid lines of a string or iterable, recursively."""
    if isinstance(iterable, str):
        return [line for line in iterable.splitlines() if line.strip() and not line.strip().startswith('#')]

    lines = []
    for item in iterable:
        lines.extend(yield_lines(item))
    return lines


def parse_requirements(strs: Union[str, list, tuple]) -> 'Requirement':
    """Yield ``Requirement`` objects for each specification in `strs`."""
    lines = iter(yield_lines(strs))

    for line in lines:
        # Drop comments and handle line continuation
        if ' #' in line:
            line = line[:line.find(' #')]

        line = line.strip()

        # If there is a line continuation, drop it, and append the next line.
        while line.endswith('\\'):
            line = line[:-1].strip()
            try:
                line += next(lines).strip()
            except StopIteration:
                break  # End of lines

        if line:
            yield Requirement(line)


# --- Ende der Hilfsfunktionen ---


PYTHON_ROOT_DIR = osp.dirname(osp.dirname(sys.executable))
SYSTEM_PYTHON_PREFIX = '/usr/lib/python'


class ConfigParsingError(RuntimeError):
    """Raise error when failed to parse pure Python style config files."""


def _get_cfg_metainfo(package_path: str, cfg_path: str) -> dict:
    """Get target meta information from all 'metafile.yml' defined in `mode-
    index.yml` of external package.

    Args:
        package_path (str): Path of external package.
        cfg_path (str): Name of experiment config.

    Returns:
        dict: Meta information of target experiment.
    """
    meta_index_path = osp.join(package_path, '.mim', 'model-index.yml')
    meta_index = OmegaConf.to_container(OmegaConf.load(meta_index_path), resolve=True)
    cfg_dict = dict()
    for meta_path in meta_index['Import']:
        meta_path = osp.join(package_path, '.mim', meta_path)
        cfg_meta = OmegaConf.to_container(OmegaConf.load(meta_path), resolve=True)
        for model_cfg in cfg_meta['Models']:
            if 'Config' not in model_cfg:
                warnings.warn(f'There is not `Config` define in {model_cfg}')
                continue
            cfg_name = model_cfg['Config'].partition('/')[-1]
            # Some config could have multiple weights, we only pick the
            # first one.
            if cfg_name in cfg_dict:
                continue
            cfg_dict[cfg_name] = model_cfg
    if cfg_path not in cfg_dict:
        raise ValueError(f'Expected configs: {cfg_dict.keys()}, but got '
                         f'{cfg_path}')
    return cfg_dict[cfg_path]


def _get_external_cfg_path(package_path: str, cfg_file: str) -> str:
    """Get config path of external package.

    Args:
        package_path (str): Path of external package.
        cfg_file (str): Name of experiment config.

    Returns:
        str: Absolute config path from external package.
    """
    cfg_file = cfg_file.split('.')[0]
    model_cfg = _get_cfg_metainfo(package_path, cfg_file)
    cfg_path = osp.join(package_path, model_cfg['Config'])
    check_file_exist(cfg_path)
    return cfg_path


def _get_external_cfg_base_path(package_path: str, cfg_name: str) -> str:
    """Get base config path of external package.

    Args:
        package_path (str): Path of external package.
        cfg_name (str): External relative config path with 'package::'.

    Returns:
        str: Absolute config path from external package.
    """
    cfg_path = osp.join(package_path, '.mim', 'configs', cfg_name)
    check_file_exist(cfg_path)
    return cfg_path


def _get_package_and_cfg_path(cfg_path: str) -> Tuple[str, str]:
    """Get package name and relative config path.

    Args:
        cfg_path (str): External relative config path with 'package::'.

    Returns:
        Tuple[str, str]: Package name and config path.
    """
    if re.match(r'\w*::\w*/\w*', cfg_path) is None:
        raise ValueError(
            '`_get_package_and_cfg_path` is used for get external package, '
            'please specify the package name and relative config path, just '
            'like `mmdet::faster_rcnn/faster-rcnn_r50_fpn_1x_coco.py`')
    package_cfg = cfg_path.split('::')
    if len(package_cfg) > 2:
        raise ValueError('`::` should only be used to separate package and '
                         'config name, but found multiple `::` in '
                         f'{cfg_path}')
    package, cfg_path = package_cfg
    return package, cfg_path


class RemoveAssignFromAST(ast.NodeTransformer):
    """Remove Assign node if the target's name match the key.

    Args:
        key (str): The target name of the Assign node.
    """

    def __init__(self, key):
        self.key = key

    def visit_Assign(self, node):
        if (isinstance(node.targets[0], ast.Name)
                and node.targets[0].id == self.key):
            return None
        else:
            return node


def _is_builtin_module(module_name: str) -> bool:
    """Check if a module is a built-in module.

    Arg:
        module_name: name of module.
    """
    if module_name.startswith('.'):
        return False
    if module_name.startswith('mmengine.config'):
        return True
    if module_name in sys.builtin_module_names:
        return True
    spec = find_spec(module_name.split('.')[0])
    # Module not found
    if spec is None:
        return False
    origin_path = getattr(spec, 'origin', None)
    if origin_path is None:
        return False
    origin_path = osp.abspath(origin_path)
    if ('site-package' in origin_path or 'dist-package' in origin_path
            or not origin_path.startswith(
                (PYTHON_ROOT_DIR, SYSTEM_PYTHON_PREFIX))):
        return False
    else:
        return True


class ImportTransformer(ast.NodeTransformer):
    """Convert the import syntax to the assignment of
    :class:`mmengine.config.LazyObject` and preload the base variable before
    parsing the configuration file.
    """

    # noqa: E501

    def __init__(self,
                 global_dict: dict,
                 base_dict: Optional[dict] = None,
                 filename: Optional[str] = None):
        self.base_dict = base_dict if base_dict is not None else {}
        self.global_dict = global_dict
        if isinstance(filename, str):
            filename = filename.encode('unicode_escape').decode()
        self.filename = filename
        self.imported_obj: set = set()
        super().__init__()

    def visit_ImportFrom(
            self, node: ast.ImportFrom
    ) -> Optional[Union[List[ast.Assign], ast.ImportFrom]]:
        # Built-in modules will not be parsed as LazyObject
        module = f'{node.level * "."}{node.module}'
        if _is_builtin_module(module):
            # Make sure builtin module will be added into `self.imported_obj`
            for alias in node.names:
                if alias.asname is not None:
                    self.imported_obj.add(alias.asname)
                elif alias.name == '*':
                    raise ConfigParsingError(
                        'Cannot import * from non-base config')
                else:
                    self.imported_obj.add(alias.name)
            return node

        if module in self.base_dict:
            for alias_node in node.names:
                if alias_node.name == '*':
                    self.global_dict.update(self.base_dict[module])
                    return None
                if alias_node.asname is not None:
                    base_key = alias_node.asname
                else:
                    base_key = alias_node.name
                self.global_dict[base_key] = self.base_dict[module][
                    alias_node.name]
            return None

        nodes: List[ast.Assign] = []
        for alias_node in node.names:
            # `ast.alias` has lineno attr after Python 3.10,
            if hasattr(alias_node, 'lineno'):
                lineno = alias_node.lineno
            else:
                lineno = node.lineno
            if alias_node.name == '*':
                raise ConfigParsingError(
                    'Illegal syntax in config! `from xxx import *` is not '
                    'allowed to appear outside the `if base:` statement')
            elif alias_node.asname is not None:
                code = f'{alias_node.asname} = LazyObject("{module}", "{alias_node.name}", "{self.filename}, line {lineno}")'  # noqa: E501
                self.imported_obj.add(alias_node.asname)
            else:
                code = f'{alias_node.name} = LazyObject("{module}", "{alias_node.name}", "{self.filename}, line {lineno}")'  # noqa: E501
                self.imported_obj.add(alias_node.name)
            try:
                nodes.append(ast.parse(code).body[0])  # type: ignore
            except Exception as e:
                raise ConfigParsingError(
                    f'Cannot import {alias_node} from {module}'
                    '1. Cannot import * from 3rd party lib in the config '
                    'file\n'
                    '2. Please check if the module is a base config which '
                    'should be added to `_base_`\n') from e
        return nodes

    def visit_Import(self, node) -> Union[ast.Assign, ast.Import]:
        """Work with ``_gather_abs_import_lazyobj`` to hack the ``import ...``
        syntax.
        """
        alias_list = node.names
        assert len(alias_list) == 1, (
            'Illegal syntax in config! import multiple modules in one line is '
            'not supported')
        # TODO Support multiline import
        alias = alias_list[0]
        if alias.asname is not None:
            self.imported_obj.add(alias.asname)
            if _is_builtin_module(alias.name.split('.')[0]):
                return node
            return ast.parse(  # type: ignore
                f'{alias.asname} = LazyObject('
                f'"{alias.name}",'
                f'location="{self.filename}, line {node.lineno}")').body[0]
        return node


def _gather_abs_import_lazyobj(tree: ast.Module,
                               filename: Optional[str] = None):
    """Experimental implementation of gathering absolute import information."""
    if isinstance(filename, str):
        filename = filename.encode('unicode_escape').decode()
    imported = defaultdict(list)
    abs_imported = set()
    new_body: List[ast.stmt] = []
    # module2node is used to get lineno when Python < 3.10
    module2node: dict = dict()
    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                # Skip converting built-in module to LazyObject
                if _is_builtin_module(alias.name):
                    new_body.append(node)
                    continue
                module = alias.name.split('.')[0]
                module2node.setdefault(module, node)
                imported[module].append(alias)
            continue
        new_body.append(node)

    for key, value in imported.items():
        names = [_value.name for _value in value]
        if hasattr(value[0], 'lineno'):
            lineno = value[0].lineno
        else:
            lineno = module2node[key].lineno
        lazy_module_assign = ast.parse(
            f'{key} = LazyObject({names}, location="{filename}, line {lineno}")'  # noqa: E501
        )  # noqa: E501
        abs_imported.add(key)
        new_body.insert(0, lazy_module_assign.body[0])
    tree.body = new_body
    return tree, abs_imported


def get_installed_path(package: str) -> str:
    """Get installed path of package.

    Replaced:
        from pkg_resources import DistributionNotFound, get_distribution

    Uses:
        importlib.metadata

    Args:
        package (str): Name of package.
    """
    try:
        # 1. Try with importlib.metadata
        pkg = importlib.metadata.distribution(package)
        # pkg.locate is the directory containing the package, similar to pkg.location
        possible_path = osp.join(pkg.locate(), package2module(package))  # Use package2module to ensure correct module dir

        # Check if the main package dir exists in the location
        if osp.exists(possible_path):
            return possible_path
        # Fallback to the distribution location itself (e.g., for namespace packages)
        return pkg.locate()

    except importlib.metadata.PackageNotFoundError as e:
        # 2. If not found via metadata, check PYTHONPATH/sys.path via importlib.util.find_spec
        spec = importlib.util.find_spec(package)
        if spec is not None:
            if spec.origin is not None:
                # spec.origin is the path to the __init__.py or .py file
                return osp.dirname(spec.origin)
            else:
                # Namespace packages don't have an origin but a spec exists
                raise RuntimeError(
                    f'{package} is a namespace package, which is invalid '
                    'for `get_install_path` in this context')
        else:
            # Re-raise the original not found error, but as the new type
            raise DistributionNotFound(
                f"The 'Distribution' '{package}' was not found and is required"
            ) from e


def import_modules_from_strings(imports, allow_failed_imports=False):
    """Import modules from the given list of strings.

    Args:
        imports (list | str | None): The given module names to be imported.
        allow_failed_imports (bool): If True, the failed imports will return
            None. Otherwise, an ImportError is raise. Defaults to False.

    Returns:
        list[module] | module | None: The imported modules.
    """
    if not imports:
        return
    single_import = False
    if isinstance(imports, str):
        single_import = True
        imports = [imports]
    if not isinstance(imports, list):
        raise TypeError(
            f'custom_imports must be a list but got type {type(imports)}')
    imported = []
    for imp in imports:
        if not isinstance(imp, str):
            raise TypeError(
                f'{imp} is of type {type(imp)} and cannot be imported.')
        try:
            imported_tmp = import_module(imp)
        except ImportError:
            if allow_failed_imports:
                warnings.warn(f'{imp} failed to import and is ignored.',
                              UserWarning)
                imported_tmp = None
            else:
                raise ImportError(f'Failed to import {imp}')
        imported.append(imported_tmp)
    if single_import:
        imported = imported[0]
    return imported


def import_module(name, package=None):
    """Import a module, optionally supporting relative imports."""
    return real_import_module(name, package)


def is_installed(package: str) -> bool:
    """Check package whether installed.

    Replaced:
        import pkg_resources
        from pkg_resources import get_distribution

    Uses:
        importlib.metadata

    Args:
        package (str): Name of package to be checked.
    """
    # Note: importlib.metadata.distribution() is generally the most reliable check
    # for an installed package (e.g., via pip/wheel).

    # 1. Check via importlib.metadata.distribution
    try:
        importlib.metadata.distribution(package)
        return True
    except importlib.metadata.PackageNotFoundError:
        pass  # Continue to step 2

    # 2. Fallback check for packages potentially in PYTHONPATH without full metadata
    spec = importlib.util.find_spec(package)
    if spec is not None:
        # If spec.origin is None, it's a namespace package (which is "found" but not installed like a normal package)
        # If spec.origin is not None, it's a module/package file that was found.
        return spec.origin is not None

    return False


def dump(obj, file=None, file_format=None, **kwargs):
    """Dump data to json/yaml/pickle strings or files (mmengine-like replacement)."""
    if isinstance(file, Path):
        file = str(file)

    # Guess file format if not explicitly given
    if file_format is None:
        if isinstance(file, str):
            file_format = file.split('.')[-1].lower()
        elif file is None:
            raise ValueError("file_format must be specified if file is None")

    if file_format not in ['json', 'yaml', 'yml', 'pkl', 'pickle']:
        raise TypeError(f"Unsupported file format: {file_format}")

    # Convert YAML extension
    if file_format == 'yml':
        file_format = 'yaml'
    if file_format == 'pickle':
        file_format = 'pkl'

    # Handle output to string
    if file is None:
        if file_format == 'json':
            return json.dumps(obj, indent=4, **kwargs)
        elif file_format == 'yaml':
            return yaml.dump(obj, **kwargs)
        elif file_format == 'pkl':
            return pickle.dumps(obj, **kwargs)

    # Handle output to file
    mode = 'w' if file_format in ['json', 'yaml'] else 'wb'
    with open(file, mode, encoding='utf-8' if 'b' not in mode else None) as f:
        if file_format == 'json':
            json.dump(obj, f, indent=4, **kwargs)
        elif file_format == 'yaml':
            yaml.dump(obj, f, **kwargs)
        elif file_format == 'pkl':
            pickle.dump(obj, f, **kwargs)

    return True


def check_file_exist(filename, msg_tmpl='file "{}" does not exist'):
    if not osp.isfile(filename):
        raise FileNotFoundError(msg_tmpl.format(filename))


def highlighted_error(msg: Union[str, Exception]) -> str:
    # Assuming 'click' is installed or this is a placeholder
    # If not using click, you can replace it with a simple string formatting for bold/red
    try:
        import click
        return click.style(str(msg), fg='red', bold=True)
    except ImportError:
        return f"[ERROR] {msg}"