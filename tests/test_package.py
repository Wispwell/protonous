import protonous


def test_every_exported_name_resolves():
    for name in protonous.__all__:
        assert getattr(protonous, name) is not None, name


def test_top_level_names_are_the_module_objects():
    from protonous.model import Model
    from protonous.tools import Toolbox

    assert protonous.Model is Model and protonous.Toolbox is Toolbox


def test_installed_version_matches_the_package():
    from importlib.metadata import version

    assert version("protonous") == protonous.__version__
