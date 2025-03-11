import kagglehub

# Download latest version
path = kagglehub.dataset_download("yassershrief/hockey-fight-vidoes")

print("Path to dataset files:", path)