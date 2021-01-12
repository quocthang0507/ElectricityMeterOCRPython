# ElectricityMeterOCRPython
 ElectricityMeterOCRPython

* To install multiple packages: `pip install -r requirements.txt`
* To backup necessary packages: `pip freeze > requirements.txt`
* To remove all packages in requirements.txt: `pip uninstall -r requirements.txt -y`
* Go to (https://pytorch.org/get-started/locally/)[https://pytorch.org/get-started/locally/], select your options or paste the script to install PyTorch (without CUDA): `pip install torch===1.7.1+cu110 torchvision===0.8.2+cu110 torchaudio===0.7.2 -f https://download.pytorch.org/whl/torch_stable.html`
[](Screenshot.png)
* Note: In the first time, you must extract .zip files in Model/
* To build executing file: `pyinstaller program_4_with_flask.py`. Executing file will be located at dist/ folder.
