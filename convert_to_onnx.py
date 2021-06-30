import torch
from models.models import Darknet

device = torch.device('cpu')
model = Darknet('cfg/yolor_csp_x.cfg', (896, 896)).cuda()
model.load_state_dict(torch.load('yolor_csp_x_star.pt', map_location=device)['model'])
model.to(device).eval()

img = torch.zeros((1, 3, 512, 896), device=device)  # init img
# Export the model
torch.onnx.export(model,               # model being run
                  img,                         # model input (or a tuple for multiple inputs)
                  "yolor_x.onnx",   # where to save the model (can be a file or file-like object)
                  export_params=True,        # store the trained parameter weights inside the model file
                  opset_version=11,          # the ONNX version to export the model to
                  do_constant_folding=True,  # whether to execute constant folding for optimization
                  input_names = ['input'],   # the model's input names
                  output_names = ['output'], # the model's output names
                  dynamic_axes={'input' : {0 : 'batch_size', 2: 'height', 3:'width'},    # Dynamic input shape
                                'output' : {0 : 'batch_size', 1: 'n_boxes'}})
print('Convert to ONNX done')
