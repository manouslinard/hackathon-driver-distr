
## Python Venv:
You should initialize the python venv in the same folder as requirements.txt (initial repo folder).<br>To create a python venv, run:
```
python3 -m venv env
```
Then to activate it:
```
source env/bin/activate
```
Once you activated the venv, you can install the requirements of the app like so (from project directory):
```
pip install -r requirements.txt
```
If you want to delete the venv, run:
```
sudo rm -rf env
```
---
## App Run:

To run the app, run:
```
python3 track.py
```