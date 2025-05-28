import os,sys, numpy as np

root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
input_dir = os.path.join(root)
output_dir = os.path.join(root, "params_int")
os.makedirs(output_dir, exist_ok=True)

dataset_names = ["mnist", "uci", "credit"]
model_names = ["linear", "svr", 'min_max_scaler']

for dataset_name in dataset_names:
    for model_name in model_names:
        params = np.load(
            os.path.join(input_dir,
                         f"{dataset_name}_{model_name}_params.npz")
        )
        int_params = {k: v.astype(int) for k, v in params.items()}
        print(f"{dataset_name}_{model_name}:")
        print(int_params)
        np.savez(
            os.path.join(output_dir,
                         f"{dataset_name}_{model_name}_params.npz"),
            **int_params
        )
print("Done!")