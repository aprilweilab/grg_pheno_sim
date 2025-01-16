# Releasing a new version

1. Update the version number in setup.py
2. Delete `dist/` to ensure no stale packages: `rm -rf dist/`
3. Build the source distribution: `python setup.py sdist`
4. Build the wheel distribution: `python setup.py bdist_wheel`
5. Push packages to PyPi: `python3 -m twine upload dist/*`
