rm -rf build/* dist/* anytrack.egg-info/*
python3 setup.py sdist bdist_wheel
twine upload dist/*
pip3 install anytrack --upgrade --user --no-cache-dir
pip3 install anytrack --upgrade --user --no-cache-dir

