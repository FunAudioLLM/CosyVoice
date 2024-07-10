## ModuleNotFoundError: No module named 'matcha'

Matcha-TTS is a third_party module. Please check `third_party` directory. If there is no `Matcha-TTS`, execute `git submodule update --init --recursive`.

run `export PYTHONPATH=third_party/Matcha-TTS` if you want to use `from cosyvoice.cli.cosyvoice import CosyVoice` in python script.

## cannot find resource.zip or cannot unzip resource.zip

Please make sure you have git-lfs installed. Execute

```sh
git clone https://www.modelscope.cn/iic/CosyVoice-ttsfrd.git pretrained_models/CosyVoice-ttsfrd
cd pretrained_models/CosyVoice-ttsfrd/
unzip resource.zip -d .
pip install ttsfrd-0.3.6-cp38-cp38-linux_x86_64.whl
```
