import copy
import time
import math
import os
import onnx
from onnx import numpy_helper
import numpy as np
import json
import traceback

from os import path
from os import listdir
from os.path import isfile, isdir, join

import torch

import tvm
import tvm.relay as relay
from tvm.contrib.download import download_testdata
from tvm import rpc
from tvm.relay import transform
from tvm.relay.dataflow_pattern import rewrite

from scipy.special import softmax

from mutators.generators import model as model_generator
from executors import mutations, libraries

from evaluators.evalgenerator import EvaluationGenerator
from loaders.model_loader import ModelLoader
from loaders.model_exporter import ModelExporter

import numpy as np
from numpy.linalg import norm

import seaborn as sns
import matplotlib.pylab as plt
import matplotlib.ticker as mtick
import pandas as pd

import re

script_dir = os.path.dirname(os.path.realpath(__file__))

def quantize(mod, params):
    with relay.quantize.qconfig(calibrate_mode="global_scale", global_scale=8.0):
        mod = relay.quantize.quantize(mod, params)
    return mod

def props(cls):
  return [i for i in cls.__dict__.keys() if i[:1] != '_']

def load_config():
    with open('./config.json') as f:
        return json.load(f)

def load_weights(path):

    onnx_model   = onnx.load(path)
    print ("Path to load: " + path)
    INTIALIZERS  = onnx_model.graph.initializer
    onnx_weights = {}
    for initializer in INTIALIZERS:
        W = numpy_helper.to_array(initializer)
        onnx_weights[initializer.name] = {"weights": W, "dims": initializer.dims}
    return onnx_weights

config = load_config()

# Common to all models configuration
device_name = config["devices"]["selected"]
build = config["devices"][device_name]
build["device_name"] = device_name
build["id"] = config["devices"]["id"]

# ----- Setup target configuration -----
target = tvm.target.Target(build["target"], host=build["host"])

# Prepare Device
host_type = build["host_type"]
device_id = build["id"]
if(host_type == "local_no_rpc"):
    remote = None
elif (host_type == "local"):
    print ("Preparing using Local RPC connection. - Device ID: " + str(device_id))
    remote = rpc.LocalSession()
else:
    address = build["address"]
    port = build["port"]
    print ("Preparing on : " + address + ":" + str(port) + " - Device ID: " + str(device_id))
    remote = rpc.connect(address, port)

datasets_info = config["datasets"]
default_dataset_info = datasets_info[config["selected_dataset"]]

images_path = script_dir + "/" + default_dataset_info["dataset_path_relative"]
image_names  = [f for f in listdir(images_path) if isfile(join(images_path, f))]

evaluation_base_folder = "/mutations/ts_full/"
evaluation_path = evaluation_base_folder + "/evaluate_mutation.txt"
evaluation_single_device_path = evaluation_base_folder + "/evaluate_single_device_mutation.txt"

mutation_model_evaluations = {}
model_names = []

opt_level = "opt" + str(config["opt_level"])


print ("Preprocessing is " + ("enabled." if config["preprocessing_enabled"] else "disabled."))

# Note: This is the raw code of the fault localization analysis.
# We intend preparing it in full and extend it.

# TODO: Move in separate class.
#------------------------- Localization Analysis -------------------------
if(config["conv_analysis_enabled"]):

    # TODO: Generate files to include.

    # TODO: Set to config file.
    debug_path1 = script_dir + config["conv_analysis_data"]["debug_path1"]
    debug_path2 = script_dir + config["conv_analysis_data"]["debug_path2"]

    # Model Parameters used for static analysis.
    orig_lib_path1 = script_dir + config["conv_analysis_data"]["lib1_activation_params"]
    orig_lib_path2 = script_dir + config["conv_analysis_data"]["lib2_activation_params"]

    analysis_path = script_dir + config["conv_analysis_data"]["analysis_output"]

    graph_dump1 = debug_path1 + "/_tvmdbg_graph_dump.json"
    graph_dump2 = debug_path2 + "/_tvmdbg_graph_dump.json"

    params1 = None
    params2 = None

    graph1 = None
    graph2 = None

    with open(orig_lib_path1, "rb") as p:
        mybytearray = bytearray(p.read())
        params1 = tvm.relay.load_param_dict(mybytearray)
        print((params1))

    with open(orig_lib_path2, "rb") as p:        
        mybytearray = bytearray(p.read())
        params2 = tvm.relay.load_param_dict(mybytearray)

    with open(graph_dump1, "r") as g:
        graph1 = json.loads(g.read())

    with open(graph_dump2, "r") as g:
        graph2 = json.loads(g.read())
  
    p_keys_to_consider1 = []
    node_to_param_map1 = {}

    p_keys_to_consider2 = []
    node_to_param_map2 = {}

    map1_nodes_arr = []
    map1_to_map2 = {}

    p_ordered = []

    #-------------------- Plot analysis (differential testing.) --------------------
    for node in graph1["nodes"]:
        for input_val in node["inputs"]:
            if (re.search("^[p][0-9]+$", input_val)):
                p_keys_to_consider1.append(input_val)
                node_to_param_map1[node["name"] + "____0"] = input_val
                map1_nodes_arr.append(node["name"] + "____0")


    count = 0
    for node in graph2["nodes"]:
        for input_val in node["inputs"]:
            if (re.search("^[p][0-9]+$", input_val)):
                p_keys_to_consider2.append(input_val)
                node_to_param_map2[node["name"] + "____0"] = input_val
                map1_tensor_name = map1_nodes_arr[count]
                map1_to_map2[map1_tensor_name] = node["name"] + "____0"
                
                count += 1

    graph_p_diffs = {}
    for input_val in p_keys_to_consider1:
        if (re.search("^[p][0-9]+$", input_val)):
            graph_p_diffs[input_val] = np.abs(params2[input_val].numpy() - params1[input_val].numpy())

    keys_keras = None
    keys_torch = None

    os.makedirs(analysis_path, exist_ok=True)

    params_files = sorted([f for f in listdir(debug_path1) if isfile(join(debug_path1, f)) and f.endswith(".params")])
    
    plot_mean_data = []
    plot_std_data = []
    plot_max_data = []
    plot_avg_data = []

    intermediate_tensors1_test = None
    intermediate_tensors2_test = None

    param_index_under_test = 0
    param_count = 0
    for param_file in params_files:

        shapes_to_explore = set()
        shapes_to_explore1 = {}
        shapes_to_explore2 = {}

        mean_data = []
        std_data = []
        max_data = []
        avg_data = []
        plot_mean_data.append(mean_data)
        plot_std_data.append(std_data)
        plot_max_data.append(max_data)
        plot_avg_data.append(avg_data)

        with open(os.path.join(debug_path1, param_file), "rb") as f:
            intermediate_tensors1 = dict(tvm.relay.load_param_dict(f.read()))
            keys_lib1 = [k for k in intermediate_tensors1.keys()]
            if(param_index_under_test == param_count):
                intermediate_tensors1_test = intermediate_tensors1

        with open(os.path.join(debug_path2, param_file), "rb") as f2:
            intermediate_tensors2 = dict(tvm.relay.load_param_dict(f2.read()))
            keys_lib2 = [k for k in intermediate_tensors2.keys()]
        
            if(param_index_under_test == param_count):
                intermediate_tensors2_test = intermediate_tensors2

        count = 0

        param_value_max_all = []
        param_value_mean_all = []
        param_value_std_all = []
        param_value_avg_all = []

        for lib1_index in node_to_param_map1.keys():
            tensor1_name = lib1_index
            
            tensor1 = intermediate_tensors1[tensor1_name]
            tensor2_name = map1_to_map2[tensor1_name]
            tensor2 = intermediate_tensors2[tensor2_name]
            
            A = tensor1.asnumpy()
            tensor1_size = 1
            for d in tensor1.shape:
                tensor1_size *= d

            param_key = node_to_param_map1[tensor1_name]
            param_value_max = np.max(graph_p_diffs[param_key])
            param_value_max_all.append(param_value_max)

            if(param_value_max == 0):
                print("Tensor found with 0 discrepancy!")
                print(tensor1_name)

            param_value_mean = np.max(graph_p_diffs[param_key])
            param_value_mean_all.append(param_value_mean)

            param_value_std = np.std(graph_p_diffs[param_key])
            param_value_std_all.append(param_value_std)

            param_value_avg = np.average(graph_p_diffs[param_key])
            param_value_avg_all.append(param_value_avg)

            tensor2_size = 1
            for d in tensor2.shape:
                tensor2_size *= d
            
            B = tensor2.asnumpy()

            res_A = np.reshape(A, (tensor1_size))
            res_B = np.reshape(B, (tensor2_size))
            dist = abs(res_A - res_B)
            mean_data.append(np.mean(dist))
            std_data.append(np.std(dist))
            max_data.append(np.max(dist))
            avg_data.append(np.average(dist))
    
        param_count += 1

    #--------------------------------------------------

    print(params_files)
    plt.rcParams["font.family"] = "Times New Roman"

    print("Params Max: " + str(np.max(param_value_max_all)))
    print("Params Avg: " + str(np.average(param_value_avg_all)))

    mean_df = {
        "Layers": np.arange(start=0, stop=len(plot_mean_data[0]), step= 1),
        "Parameters": param_value_mean_all
    }

    max_df = {
        "Layers": np.arange(start=0, stop=len(plot_max_data[0]), step= 1),
        "Parameters": param_value_max_all
    }

    std_df = {
        "Layers": np.arange(start=0, stop=len(plot_std_data[0]), step= 1),
        "Parameters": param_value_std_all
    }

    i = 0
    for param_name in params_files:
        param_name = "Image " + str(i + 1)
        mean_df[param_name] = plot_mean_data[i]
        max_df[param_name] = plot_max_data[i]
        std_df[param_name] = plot_std_data[i]
        i += 1

    mean_data_frame = pd.DataFrame(mean_df)
    max_data_frame = pd.DataFrame(max_df)
    std_data_frame = pd.DataFrame(std_df)
    mean_data_frame.to_csv(analysis_path + "/mean_df.csv")
    std_data_frame.to_csv(analysis_path + "/std_df.csv")
    max_data_frame.to_csv(analysis_path + "/max_df.csv")

    max_differences = {
    }

    for count_index in range(len(params_files)):
        params_file = params_files[count_index]
        for check_index in range(count_index + 1, len(params_files)):

            detect_max_diff = abs(np.array(plot_max_data[count_index]) - np.array(plot_max_data[check_index]))
            detect_max_diff = sorted(enumerate(detect_max_diff), key=lambda i: i[1], reverse=True)
            params_file_check = params_files[check_index]

            key = params_file.replace(".params", "") + "_vs_" + params_file_check.replace(".params", "")
            max_differences[key] = []
            for i in range(10):
                tuple_max = detect_max_diff[i]
                max_index = tuple_max[0]
                tensor1_name = map1_nodes_arr[max_index]
                tensor2_name = map1_to_map2[tensor1_name]
                max_differences[key].append({
                    "model1_layer_name": tensor1_name,
                    "model2_layer_name": tensor2_name,
                    "abs_difference": str(tuple_max[1])
                })

    json_object = json.dumps(max_differences, indent=2)
    
    with open(analysis_path + "diff_across_images.json", "w") as outfile:
        outfile.write(json_object)

    tuple_max = detect_max_diff[0]
    plt.figure(figsize=(7.5, 4.5), dpi=600)
    ax = sns.lineplot(data=pd.melt(mean_data_frame, "Layers"), linewidth=1, \
        x="Layers", y="value", hue="variable", err_style="bars", palette='magma')
    ax.yaxis.grid(b=True, which='major', color='black', linewidth=0.075)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles[0:], labels=labels[0:], title="")
    plt.xlabel("Layers")
    plt.ylabel("Difference")

    plt.savefig(os.path.join(analysis_path + "", "layers_mean.png"))

    plt.close()

    plt.figure(figsize=(7.5, 4.5), dpi=600)
    ax = sns.lineplot(data=pd.melt(std_data_frame, "Layers"), linewidth=1, x="Layers", \
        y="value", hue="variable", err_style="bars", palette='magma')
    ax.yaxis.grid(b=True, which='major', color='black', linewidth=0.075)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles[0:], labels=labels[0:], title="")
    plt.xlabel("Layers")
    plt.ylabel("Difference")

    plt.savefig(os.path.join(analysis_path + "", "layers_std.png"))
    plt.close()

    plt.figure(figsize=(7.5, 4.5), dpi=600)
    ax = sns.lineplot(data=pd.melt(max_data_frame, "Layers"), linewidth=1, x="Layers", \
        y="value", hue="variable", err_style="bars", palette='magma')
    ax.yaxis.grid(b=True, which='major', color='black', linewidth=0.075)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles[0:], labels=labels[0:], title="")
    plt.xlabel("Layers")
    plt.ylabel("Difference")

    plt.savefig(os.path.join(analysis_path + "", "layers_max.png"))
    plt.close()

model_loader = ModelLoader()
model_exporter = ModelExporter()

mutations_executor = None

#------------------------- Models Processing -------------------------

for loop_count in range(config["runs_no"]):
    print("Run " + str(loop_count + 1) + " out of " + str(config["runs_no"]) + ".")
    for model_info in config["models"]:

        if("skip_analysis" in model_info and model_info["skip_analysis"]):
            print("Skipping analysis on " + model_info["alias"])
            continue

        print("Processing model " + model_info["alias"])

        model_url = model_info["url"]
        input_name = model_info["input_name"]
        output_name = model_info["output_name"] if "output_name" in model_info else None
        paths_info = model_info["paths"]
        models_path = script_dir + "/" + paths_info["models_out_relative"]
        generated_path = script_dir + "/generated/"

        input_layer_shape = tuple(model_info["layers"]["input"])
        x = np.zeros(shape=input_layer_shape)
        shape_dict = {input_name: x.shape}

        out_path = script_dir + paths_info["exec_out_relative"]
        dll_out_path = script_dir + paths_info["dll_exec_out_relative"]
        error_base_folder = join(script_dir, "error_log", model_info["name"])


        models_data = {
            "name": model_info["name"],
            "model": model_info["model"],
            "model_name": model_info["name"],
            "raw_model_name": model_info["name"],
            "input_model_folder": models_path,
            "output_model_folder": models_path,
            "image_dimension": tuple(model_info["layers"]["image_dimension"]),
            "input": tuple(model_info["layers"]["input"]),
            "input_name": input_name,
            "output": tuple(model_info["layers"]["output"]),
            "output_name": output_name,
            "library": model_info["library"] if "library" in model_info else "unknown",
            "preprocessing_enabled": config["preprocessing_enabled"],
            "debug_enabled": config["debug_enabled"],
            "dll_models_path": models_path,
            "dtype": model_info["dtype"] if "dtype" in model_info else "float32"
        }

        # ----- Model building & mutations phase -----
        if(loop_count == 0 and ("build" in model_info and model_info["build"])):
            print("Building: " + model_info["name"])
            os.makedirs(models_path, exist_ok=True)

            # ----- Model Download and Loading Phase -----
            if ("type" in model_info and model_info["type"] == "library"):
                mod, params = model_loader.load_model(models_data)
            else:
                if ("type" in model_info and model_info["type"] == "remote"):
                    model_path = script_dir + "/" + model_url
                else:
                    model_path = download_testdata(model_url, model_info["alias"], module=model_info["type"])
                onnx_model = onnx.load(model_path)
                mod, params = relay.frontend.from_onnx(onnx_model, shape_dict)
                
            if("quantize" in config and config["quantize"]):
                print("Quantization enabled.")
                mod = quantize(mod, params)

            model_generator.generate_original_model(mod, target, params, paths_info["models_out_relative"], opt_level=config["opt_level"], required_pass=config["required_pass"], disabled_pass=config["disabled_pass"], quantize=config["quantize"], opt_alias=config["opt_alias"])
            
            if ("enable_mutations" in config and config["enable_mutations"]):
                if ("positions" in mutations_info):
                    info = mutations_info["positions"][model_info["name"]]
                    relay_positions = info["relay"]
                    for relay_position in relay_positions:
                        for mutation in mutations_info["relay"]:
                            mutation["start"] = relay_position
                            mutation["end"] = relay_position
                        model_generator.generate_relay_mutations(mod, target, params, mutations_info["relay"], paths_info["models_out_relative"], opt_level=config["opt_level"], required_pass=config["required_pass"], disabled_pass=config["disabled_pass"], quantize=config["quantize"], opt_alias=config["opt_alias"])
                        
                    tir_positions = info["tir"]
                    for tir_position in tir_positions:
                        for mutation in mutations_info["tir"]:
                            mutation["start"] = tir_position
                            mutation["end"] = tir_position
                        model_generator.generate_tir_mutations(mod, target, params, mutations_info["tir"], paths_info["models_out_relative"], opt_level=config["opt_level"], required_pass=config["required_pass"], disabled_pass=config["disabled_pass"], quantize=config["quantize"], opt_alias=config["opt_alias"])
            
                else:
                    model_generator.generate_relay_mutations(mod, target, params, mutations_info["relay"], paths_info["models_out_relative"], opt_level=config["opt_level"], required_pass=config["required_pass"], disabled_pass=config["disabled_pass"], quantize=config["quantize"], opt_alias=config["opt_alias"])
                    model_generator.generate_tir_mutations(mod, target, params, mutations_info["tir"], paths_info["models_out_relative"], opt_level=config["opt_level"], required_pass=config["required_pass"], disabled_pass=config["disabled_pass"], quantize=config["quantize"], opt_alias=config["opt_alias"])

        mutations_names = []
        if (path.exists(models_path)):
            mutations_names = [f for f in listdir(models_path) if isfile(join(models_path, f)) and f.endswith(".tar") and ("_ignore_" not in f)]

        mutations_names.sort(key=lambda x: "original" in x, reverse=True)
        print("Mutations generated:")
        print(mutations_names)
        models_data["mutations_names"] = mutations_names
        dll_models_path = script_dir + "/" + paths_info["dll_models_out_relative"]
        dll_libs = model_info["dll_libraries"]
        
        if(loop_count == 0 and ("build_dlls" in model_info and model_info["build_dlls"])):
            
            print("Building DL models for " + model_info["name"])
            os.makedirs(dll_models_path, exist_ok=True)
            
            for dll_lib in dll_libs:
                if("noop" not in dll_lib["library"]):
                    dll_lib["script_dir"] = script_dir
                    dll_lib["name"] = model_info["name"]
                    dll_lib["dtype"] = model_info["dtype"] if "dtype" in model_info else "float32"
                    dll_lib["dll_models_path"] = dll_models_path

                    dll_model, dll_params = model_loader.load_model(dll_lib)
                    print("MODEL LOADED!")


                    if("quantize" in config and config["quantize"]):
                        print("Quantization enabled.")
                        dll_model = quantize(dll_model, dll_params)
                    if(dll_model != None):
                        model_generator.generate_original_model(dll_model, target, dll_params, paths_info["dll_models_out_relative"], (dll_lib["model"] + "_" + dll_lib["library"]).replace(".", "_"), opt_level=config["opt_level"], required_pass=config["required_pass"], disabled_pass=config["disabled_pass"], quantize=config["quantize"], opt_alias=config["opt_alias"])


        if(loop_count == 1 and "export_dlls" in model_info and model_info["export_dlls"]):

            print("Exporting DL models for " + model_info["name"])
            export_dll_models_path = script_dir + "/" + paths_info["export_dll_models_out_relative"] if "export_dll_models_out_relative" in paths_info else dll_models_path
            os.makedirs(export_dll_models_path, exist_ok=True)
            export_dll_libs = model_info["export_dll_libraries"]

            for export_dll_lib in export_dll_libs:
                if("noop" not in export_dll_lib["library"]):
                    export_dll_lib["script_dir"] = script_dir
                    export_dll_lib["name"] = model_info["name"]
                    export_dll_lib["dtype"] = model_info["dtype"] if "dtype" in model_info else "float32"
                    export_dll_lib["export_dll_models_path"] = export_dll_models_path

                    model_exporter.export_model(export_dll_lib)

        folders_to_execute = []
        folders_to_execute.append(images_path)

        images_data = {
            "input_images_folders": folders_to_execute,
            "output_images_base_folder": out_path
        }

        build["error_base_folder"] = error_base_folder

        # -----Models execution phase-----
        try:
            # Direct NN model Execution from ONNX.
            if("execute" in model_info and model_info["execute"]):

                print("Executing: " + model_info["name"])

                if mutations_executor is None:
                    mutations_executor = mutations.MutationsExecutor(models_data, images_data, build)
                mutations_executor.execute(remote)

            # Execution of DLLs.
            if("execute_dlls" in model_info and model_info["execute_dlls"]):
                
                dll_models_data = models_data.copy()
                dll_models_data["input_model_folder"] = dll_models_path
                dll_models_data["output_model_folder"] = dll_models_path

                dll_images_data = images_data.copy()
                dll_images_data["output_images_base_folder"] = dll_out_path

                # Run with all given libraries.
                for dll_lib in dll_libs:
                    if("noop" not in dll_lib["library"]):
                        print("Executing: " + model_info["name"] + "(" + dll_lib["library"] + ")")

                        # TODO: Refactor this naming convention, it is very confusing!
                        dll_models_data["name"] = model_info["name"]
                        dll_models_data["model_name"] = dll_lib["dependency"]
                        dll_models_data["converted_lib_model_name"] = dll_lib["converted_lib_model_name"] if "converted_lib_model_name" in dll_lib else None
                        dll_models_data["raw_model_name"] = dll_lib["model"]
                        dll_model_name = (dll_lib["model"] + "_" + dll_lib["library"])
                        dll_model_name = dll_model_name.replace("_", "{DASH}").replace(".", "_").replace("{DASH}", "_")
                        dll_model_name = dll_model_name + ("_quant" if ("quantize" in config and config["quantize"]) else "")
                        dll_model_name = dll_model_name + "_opt" + str(config["opt_level"]) 
                        dll_model_name = dll_model_name + config["opt_alias"] + ".tar"
                        dll_models_data["mutations_names"] = [dll_model_name]
                        dll_input_name = dll_lib["input_name"]
                        dll_output_name = dll_lib["output_name"]
                        dll_models_data["input_name"] = dll_input_name
                        dll_models_data["output_name"] = dll_output_name
                        dll_models_data["input"] = dll_lib["input"]
                        dll_models_data["library"] = dll_lib["library"]
                        dll_models_data["output"] = dll_lib["output"]
                        dll_models_data["args_no"] = dll_lib["args_no"] if "args_no" in dll_lib else 1

                        if("image_dimension" in dll_lib):
                            dll_models_data["image_dimension"] = tuple(dll_lib["image_dimension"])
                        else:
                            dll_models_data["image_dimension"] = tuple(model_info["layers"]["image_dimension"])

                        if (config["backend"] != "tvm" or
                            ("execution_type" in dll_lib and dll_lib["execution_type"] == "library")):
                            libraries_executor = libraries.LibrariesExecutor(dll_models_data, dll_images_data, build)
                            libraries_executor.execute()
                        else:
                            mutations_executor = mutations.MutationsExecutor(dll_models_data, dll_images_data, build)
                            mutations_executor.execute(remote)
            
        except Exception as e:
            print(traceback.print_exc())

        # Evaluate executions.
        if ("evaluate" in model_info and model_info["evaluate"]):
            model_base = script_dir + "/" + paths_info["evaluation_out_relative"]
            mutations_names = [f for f in listdir(model_base) if isfile(join(model_base, f)) and f.endswith(".tar") and ("_ignore_" not in f)]
            generated_models_prettified = [model.replace(".tar", "").replace(".", "_") for model in mutations_names]
            eg_mts = EvaluationGenerator()
            
            print("Evaluating: " + model_info["name"])

            device_index = 0
            device_folders = [d for d in listdir(model_base) if isdir(join(model_base, d))]

            # Compare everything.
            for device_folder in device_folders:
                print("Device Folder:" + device_folder)
                eg_mts.generate_libraries_comparison(join(model_base, device_folder), device_folder=device_folder) #base_case_folder="model_original_opt4"
                eg_mts.generate_base_folder_comparison(generated_models_prettified, join(model_base, device_folder), "mutations")
                eg_mts.generate_devices_comparison(model_base, replace_evaluated_suffix=True)
                eg_mts.get_time_stats_folder(join(model_base, device_folder))
