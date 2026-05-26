def test_package_import_has_version():
    import kalorie

    assert isinstance(kalorie.__version__, str)
    assert kalorie.__version__
