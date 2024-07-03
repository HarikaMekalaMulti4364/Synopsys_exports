from sne4onnx import extraction

extracted_graph = extraction(
  input_op_names=['input_mask'],
  output_op_names=['400'],
  input_onnx_file_path="/media/ava/DATA3/DATA/Harika/random/inference/language/bert/build/data/bert_tf_v1_1_large_fp32_384_v2/model.onnx",
  output_onnx_file_path='model_400.onnx',
)

import numpy as np
input_ids = np.load("/media/ava/DATA3/DATA/Harika/random/inference/language/bert/input_ids1.npy")
input_ids = np.expand_dims(input_ids, axis=0)
segment_ids = np.load("/media/ava/DATA3/DATA/Harika/random/inference/language/bert/segment_ids1.npy")
segment_ids = np.expand_dims(segment_ids, axis=0)
input_mask = np.load("/media/ava/DATA3/DATA/Harika/random/inference/language/bert/input_mask1.npy")
input_mask = np.expand_dims(input_mask, axis=0)
print(segment_ids.shape)
onnx_input = {
    # "input_ids" : input_ids,
    # "segment_ids" : segment_ids
    "input_mask" : input_mask
    }
import onnx
import onnxruntime as ort
model_path = "/media/ava/DATA3/DATA/Harika/random/inference/language/bert/model_400.onnx"
model = onnx.load(model_path)
session = ort.InferenceSession(model_path)
output = session.run(None, onnx_input)
file_name = "model_400_output.npy"
np.save(file_name, output[0])