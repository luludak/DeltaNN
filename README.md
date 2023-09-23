# DeltaNN

DeltaNN is a comprehensive suite for compiling, optimizing, executing and analyzing pretrained DNNs under different computational environment settings.

*This is the project of the publication ["DeltaNN: Assessing the Impact of Computational Environment Parameters on the Performance of Image Recognition Models"](https://arxiv.org/abs/2306.06208), to be presented at IEEE ICSME 2023.
Related results, data and supplementary material can be found [here](https://github.com/luludak/DeltaNN-Results).*

In total, the suite supports:
- Build of Neural Networks using different backend DL Frameworks.
- Build of DNNs under different optimization settings.
- Build of DNNs using different GPU backends (CUDA, OpenCL, etc).
- Conversion of DNNs from one backend framework to another (currently supporting all conversions across Keras, PyTorch, TF, TFlite).
- Executing DNNs in different hardware acceleration environments.
- Analyzing the results in a bundled and automated manner.
- Providing activation maps localization analysis (Alpha vesion).

The suite is based on [Apache TVM](https://tvm.apache.org/) for its capabilities.

## Installation

The system needs TVM to be installed.
We also use `Python v3.8.5` and `Pip` as the package installer.

In addition, the system requires a number of pip packages, which you can find in the requirements.txt file.

## Instructions:

1. Install Python and Pip on your system.
- Python comes with linux distros usually, but this is not always the case for Pip. You can install it by running `sudo apt install python3-pip`.
2. Download and install TVM:
For instructions of how to install TVM, please refer to the [TVM related guide for developers](https://tvm.apache.org/docs/install/from_source.html#developers-get-source-from-github).
Follow the installation from source instructions, and consider enabling the LLVM and the OPENCL flags.

3. Install necessary packages by executing the command:
`pip3 install -r requirements.txt`

4. Download necessary TF/TFLite models, if you wish to run them.
Although system utilizes already provided models for Keras and PyTorch, we utilized some TF/TFlite models from the GitHub repo of Tensorflow for slim Models. These are:
- `MobileNetV2`
- `ResNet101`
- `InceptionV3`

You can download them manually and place them in the models folder of each model from [the official TensorFlow repo](https://github.com/tensorflow/models/tree/master/research/slim).

Following download, extract and put the models into `<script_folder>/generated/<Model_Folder>/models` folder. Do so for both .pb and .tflite models.
Also, make sure the names of the models are the same as in configuration for each model.

## Configuration
The configuration of the system is included into the config.json file.
Each section is self-explanatory and defines which part it concerns.
Important notes:
- You can run the models **without** TVM, directly using the library of your choice. In this case, set the flag `backend` to `libraries` instead of `tvm`.
- You can utilize the TVM debugger, by setting `debug_enabled: true`.
- `build` and `execute` flags concerns the ONNX model defined in the URL and will apply actions only to this. If you want DLLs to be built or executed, mark flag `build_dlls` or `execute_dlls` as true.
- `evaluate` flag concerns DLLs as well.
- Device settings have been cleared out to preserve anonymity. If you wish, you can set up your own TVM RPC server on your own device and run everything following the instructions [here](
https://tvm.apache.org/docs/tutorial/cross_compilation_and_rpc.html).

## Example
In order to verify your installation and be able to run the framework with your own configuration, we have setup the configuration to build the system utilizing 3 libraries:
1. TFLite (Downloaded an included from the TF repo aforementioned).
2. Keras (Using pre-built library of keras).
3. PyTorch (Same as keras).

As Dataset, we provide a small dataset, obtained from [unsplash](https://unsplash.com/images/stock/public-domain). No copyright infingement intended.
We provide the native TF and TFLite models, obtained from [TensorFlow zoo slim repo](https://github.com/tensorflow/models/tree/master/research/slim/), while the system supports inference and conversion across the pretrained models that are part of the Keras and PyTorch DL frameworks API.

Once you set up the framework, you can execute it by doing:
`python3 main.py`

The example case will build, run and execute evaluation for `MobileNetV2`, in `TFLite` DL Framework. The evaluation will give an empty devices file, as no simultaneous library runs are performed, and there are no other runs to additional devices.

#### Build: 
The system will generate the models in the folder defined in config.json, along with their generated Host/Kernel code, but also their TVM Relay IR code:
`<script_folder>/generated/MobileNet-2-7/models`

In total, the framework will generate the models compiled on TVM, utilizing the `opt=2` optimization setting, to be executed using `OpenCL` for hardware acceleration, for `TFLite`, `Keras` and `PyTorch`.

##### Convert:
DeltaNN supports conversions of DL frameworks, for Keras, PyTorch, TF, TFlite. This can be enabled by setting <source>_to_<target> model in `dll_libraries` configuration of a model, in `config.json` file. For Keras, add `keras_library` as source.

That way, the suite will utilize PyTorch model and apply conversion to generate a the respective target model from source. The model then will be treated as a first-class model citizen that can be built under different optimization and hardware acceleration settings, be executed and analyzed.


#### Execute:
Your system will then execute, generating a folder with experiments. The structure is:
`<script_folder>/<generated>/<model>/<device>/<opt_level>/`
In this case:
`<script_folder>/out_large/local/opt2/<library_folders>/mutations/ts_<epoch_time_of_run>/<predictions>.txt`

Each file, will contain the top-5 predictions, along with the execution time per-prediction at the bottom.
In addition, you will find an execution_log.txt file in the aforementioned fonder, containing info about the run.

Console will indicate the status of the running model and update accordingly.

#### Analyze:
Once execution is complete, analysis will be executed. This will be done in 3 ways:
- Comparing results per-device (if provided).
- Comparing results per-library(if provided).
- Comparing results per-multiple executions (if provided).

The system will then generate the following files:
`<script_folder>/<generated>/<model>/<device>/device_evaluation.json` containing results per-device comparison in a N-N manner.
`<script_folder>/<generated>/<model>/<device>/<opt_level>/library_evaluation.json` containing results per-library comparison in a N-N manner.
`<script_folder>/<generated>/<model>/<device>/<opt_level>/library_evaluation_time_percentage.json` containing percentages of execution time relative change per-prediction across libraries.
`<script_folder>/<generated>/<model>/<device>/<opt_level>/mutations/same_folder_evaluation.json`, containing the comparison across multiple executions.

Notice that there is a shift across TF/TFLite libraries and Keras/PyTorch. For example, for an image of drums, in the case of correct classification, TFLite will give the ImageNet ID:`542`, while the other two libraries, `541`. This was consistent behaviour across library executions, and we considered the offset to our comparisons. If you observe the analysis files, you will see that the comparison output is 100% across all libraries. However this does not apply in the comparison across source and target libraries upon conversion.

 To perform a full comparison, set the files to run in structure:

`<optimization>/<device>/<dl_framework>` and then update the `evaluation_out_relative` value of the model to contain the optimization setting under test.

For example, if you have run the experiments for 2 optimizations, 2 devices and 2 libraries with 1 conversion, your structure should be:

```
.
├── Opt0/
│   ├── Device 1/
│   │   ├── TF
│   │   ├── TFLite
│   │   └── TF-To-TFLite
│   └── Device 2/
│       ├── TF
│       ├── TFLite
│       └── TF-To-TFLite
└── Opt2/
    ├── Device 1/
    │   ├── TF
    │   ├── TFLite
    │   └── TF-To-TFLite
    └── Device 2/
        ├── TF
        ├── TFLite
        └── TF-To-TFLite
```
        
And then set `"evaluation_out_relative": <root_folder>/<Opt0/2>`, depending on the optimization setting under analysis.

#### Localize Faults (Alpha version):
The system includes a mechanism for fault localization. By setting `conv_analysis_enabled=true` in the config.json file, the system will consider two model metadata in order to perform fault localization. For that matter, the system will need (1) the variants of models built on TVM, and (2) inference of images presenting different results across source and target models, in TVM debug mode (having generated debugger "params" metadata). The data required must be provided in the config file.

The system performs layer activations and parameters analysis and comparison, and generates plots with the respective data.
Note that this version of the system needs refactoring, and is in alpha version.


#### Errors:
In case of an error, the suite will generate a `<script_folder>/error_log/<model>/ts_<epoch_time_of_problematic_run>/error_log.txt` file containing all related info.

#### Alpha Features:
Inside `main.py`, you will also find the alpha version of specific features, such as:
- Neural Network layers activation localization (Pre-Alpha Version).
- Plot Generation for execution times across devices, DL Frameworks, as well as comparisons of output predictions for DL Frameworks (Alpha Version).

You can enable those features manually, however we intend fully integrating them to our system.

As a last note, you can try your own model, given you provide the right files and settings. Configuration provides exactly the details requested for a model to be loaded from a backend, compiled using a specific optimization and GPU backend and be run for inference, respectfully.


#### CLOC (Excluding External Folder)
```
Language                     files          blank        comment           code
-------------------------------------------------------------------------------
Python                          13            703            249           2030
JSON                             1              2              0            829
Markdown                         1             41              0            113
-------------------------------------------------------------------------------
SUM:                            15            746            249           2972
-------------------------------------------------------------------------------
```
