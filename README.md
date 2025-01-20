### install the package
`pip install --upgrade git+https://github.com/sleepingcat4/deep-enhance.git`

### Downloading the model
```Python
from resemble_enhance.enhancer import download

run_dir = "/content/model" # defaults to model
download.download(run_dir)
```
**Note:** Download the model by default in *model* name folder. If not present it will be created. 

### Running inference
```Python
from resemble_enhance.enhancer import inference_terminal

input_folder = "/exp/input_folder"
output_folder = "/exp/output_folder"
inference_terminal.enhance_folder(input_folder, output_folder, solver="midpoint", nfe=64, tau=0.5, run_dir="/exp/model")
```

**Note:** Provide path where you had saved the model and input and output folder paths. We designed it to work on folders as we wanted to knock-down large amounts of files with little to no effort. On T4 GPU, We can finish enhancement of 4 files each (8 seconds in length) in 15.14 seconds. Batch inference is unfortunately not possible in the current implementation. It requires more advanced modifications. 

### Acknowledge
We recognise the rights and contributions of Resemble ai team. This is a fork of their original repository made it to work on EU supercomputers. 
