import os
import json
import paramiko
from paramiko.proxy import ProxyCommand  # Import ProxyCommand
from glob import glob
import fnmatch
import stat

# Define a list of subdirectories
subdirectories = [
    # "cnn_dailymail",
    # "pile_StackExchange",
    # "pile_Pile-CC", 
    # "pile_Wikipedia (en)",
    "pile_USPTO Backgrounds",
    # "pile_Ubuntu IRC",
    # "pile_PhilPapers", 
    # "pile_EuroParl", 
    # "pile_NIH ExPorter",
    # "pile_HackerNews"
    # "pile_PubMed Central",
    # "pile_PubMed Abstracts",
    "pile_Enron Emails",
    # "pile_ArXiv",
    # "pile_Github",
    # "pile_FreeLaw",
    "pile_DM Mathematics",
    # "dolma-v1_7_cc",
    # "dolma-v1_7_reddit",
    # "dolma-v1_7_wiki",
    # "dolma-v1_7_books",
    # "dolma-v1_7_pes2o",
    # "dolma-v1_7_stack",
]
# subdirectories = [
#     "timothy_sykes"]

# Define base local and remote paths
base_local_path = "/storage2/bihe/llm_data_detect/datasets"
base_remote_path = "/home/c01bizh/CISPA-home/data/llm_data_detect/datasets"

base_local_path_model = "/storage2/bihe/llm_data_detect/model"
base_remote_path_model = "/home/c01bizh/CISPA-home/data/llm_data_detect/model"

# Define file patterns

upload_file_patterns, download_file_patterns, download_model_patterns = [], [], []

for subset_name in subdirectories:
    dataset = '_'.join(subset_name.split('_')[:-1])
    subset = subset_name.split('_')[-1]
    if dataset == 'pile':
        config_file_path = '/home/bihe/LLM_data_detect/subset_config_search.json'
    elif dataset.startswith('dolma'):
        config_file_path = '/home/bihe/LLM_data_detect/subset_config_search_dolma.json'
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
    with open(config_file_path, 'r') as file:
        config_dict = json.load(file)
    gen_configs = config_dict[subset]["gen_configs"]
    for gen_config in gen_configs:
        doc_idx = gen_config["doc_idx"]
        max_snippets = gen_config["max_snippets"]
        n_tokens_list = gen_config["n_tokens_list"]
        for n_token in n_tokens_list:
            # for uploading
            upload_file_patterns.append(f"train_{doc_idx}_{n_token}token_max{max_snippets}*")
            upload_file_patterns.append(f"val+test_{doc_idx}_{n_token}token_max{max_snippets}*")
            # for downloading
            # download_file_patterns.append(f"train_{doc_idx}_{n_token}token_max{max_snippets}*")
            # download_file_patterns.append(f"val+test_{doc_idx}_{n_token}token_max{max_snippets}*")
            # download_file_patterns.append(f"train_{doc_idx}_{n_token}token_max{max_snippets}*_1.jsonl")
            # download_file_patterns.append(f"train_{doc_idx}_{n_token}token_max{max_snippets}*_100.jsonl")
            # download_file_patterns.append(f"val+test_{doc_idx}_{n_token}token_max{max_snippets}*_1.jsonl")
            # download_file_patterns.append(f"val+test_{doc_idx}_{n_token}token_max{max_snippets}*_100.jsonl")
            download_file_patterns.append(f"*.jsonl")
            # for downloading model checkpoints
            # download_model_patterns.append(f"*train_{doc_idx}_{n_token}token_max{max_snippets}*")
            # download_model_patterns.append(f"*val+test_{doc_idx}_{n_token}token_max{max_snippets}*")
            download_model_patterns.append(f"*")

# download_model_patterns.append(f"*")

# Load SSH configuration
ssh_config_path = os.path.expanduser("~/.ssh/config")
ssh_config = paramiko.SSHConfig()
with open(ssh_config_path, "r") as f:
    ssh_config.parse(f)

host_config = ssh_config.lookup("cispa")
hostname = host_config.get("hostname")
port = int(host_config.get("port", 22))
username = host_config.get("user")
private_key_path = host_config.get("identityfile", [None])[0]

# Manually construct ProxyCommand if missing
proxy_command = host_config.get("proxycommand")
if not proxy_command and "proxyjump" in host_config:
    proxy_jump = host_config["proxyjump"]
    proxy_command = f"ssh -W {hostname}:{port} {proxy_jump}"

# Debugging information
print(f"Resolved SSH Config:\nHost: {hostname}\nPort: {port}\nUser: {username}\nProxyCommand: {proxy_command}")

# Function to upload files
def upload_files(sftp):
    for subdir in subdirectories:
        local_path = os.path.join(base_local_path, subdir)
        remote_path = os.path.join(base_remote_path, subdir)
        try:
            sftp.listdir(remote_path)
        except FileNotFoundError:
            print(f'no such path on remote server: {remote_path}. Creating..')
            sftp.mkdir(remote_path)
        for pattern in upload_file_patterns:
            files_to_upload = glob(os.path.join(local_path, pattern))
            for file_path in files_to_upload:
                remote_file_path = os.path.join(remote_path, os.path.basename(file_path))
                print(f"Uploading {file_path} to {remote_file_path}...")
                sftp.put(file_path, remote_file_path)

# Function to download files
# def download_files(sftp):
#     for subdir in subdirectories:
#         local_path = os.path.join(base_local_path, subdir)
#         remote_path = os.path.join(base_remote_path, subdir)
#         # os.makedirs(local_path, exist_ok=True)
#         try:
#             remote_files = sftp.listdir(remote_path)
#         except FileNotFoundError:
#             print(f"Remote directory {remote_path} does not exist.")
#             continue
#         for file_name in remote_files:
#             for pattern in download_file_patterns:
#                 if fnmatch.fnmatch(file_name, pattern):
#                     remote_file_path = os.path.join(remote_path, file_name)
#                     local_file_path = os.path.join(local_path, file_name)
#                     print(f"Downloading {remote_file_path} to {local_file_path}...")
#                     sftp.get(remote_file_path, local_file_path)

def download_files(sftp, base_remote_path=None, base_local_path=None, subdirectories=None, download_file_patterns=None):
    if subdirectories is None:
        subdirectories = [""]
    
    for subdir in subdirectories:
        local_path = os.path.join(base_local_path, subdir)
        remote_path = os.path.join(base_remote_path, subdir)
        os.makedirs(local_path, exist_ok=True)
        
        try:
            remote_items = sftp.listdir(remote_path)
        except FileNotFoundError:
            print(f"Remote directory {remote_path} does not exist.")
            continue
            
        for item_name in remote_items:
            remote_item_path = os.path.join(remote_path, item_name)
            local_item_path = os.path.join(local_path, item_name)
            
            # Check if item is a directory
            try:
                item_stat = sftp.stat(remote_item_path)
                is_dir = stat.S_ISDIR(item_stat.st_mode)
            except:
                print(f"Could not determine if {remote_item_path} is a directory.")
                continue
                
            if is_dir:
                # Check if directory matches any pattern
                for pattern in download_file_patterns:
                    if fnmatch.fnmatch(item_name, pattern):
                        print(f"Found matching directory: {remote_item_path}")
                        # Recursively download the directory
                        download_directory_recursively(sftp, remote_item_path, local_item_path)
                        break
            else:
                # File handling - same as before
                for pattern in download_file_patterns:
                    if fnmatch.fnmatch(item_name, pattern):
                        print(f"Downloading {remote_item_path} to {local_item_path}...")
                        sftp.get(remote_item_path, local_item_path)
                        break

def download_directory_recursively(sftp, remote_dir, local_dir):
    """Recursively download all files and subdirectories from remote_dir to local_dir"""
    os.makedirs(local_dir, exist_ok=True)
    
    try:
        items = sftp.listdir(remote_dir)
    except FileNotFoundError:
        print(f"Remote directory {remote_dir} does not exist.")
        return
        
    for item_name in items:
        remote_item_path = os.path.join(remote_dir, item_name)
        local_item_path = os.path.join(local_dir, item_name)
        
        try:
            item_stat = sftp.stat(remote_item_path)
            is_dir = stat.S_ISDIR(item_stat.st_mode)
        except:
            print(f"Could not determine if {remote_item_path} is a directory.")
            continue
            
        if is_dir:
            # Recursively download subdirectory
            download_directory_recursively(sftp, remote_item_path, local_item_path)
        else:
            # Download file
            print(f"Downloading {remote_item_path} to {local_item_path}...")
            sftp.get(remote_item_path, local_item_path)

# Connect to the SFTP server
try:
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    # Handle ProxyCommand
    proxy = None
    if proxy_command:
        print(f"Using ProxyCommand: {proxy_command}")
        proxy = ProxyCommand(proxy_command)

    private_key = paramiko.RSAKey.from_private_key_file(private_key_path) if private_key_path else None
    print(f"Connecting to {hostname}:{port} as {username}...")
    ssh.connect(hostname, port, username, pkey=private_key, sock=proxy)

    sftp = ssh.open_sftp()

    action = input("Enter 'upload' to upload files or 'download' to download files: ").strip().lower()
    if action == "upload":
        upload_files(sftp)
    elif action == "download":
        # download data
        download_files(sftp, base_remote_path=base_remote_path, base_local_path=base_local_path, subdirectories=subdirectories, download_file_patterns=download_file_patterns)
        # download model checkpoints
        download_files(sftp, base_remote_path=base_remote_path_model, base_local_path=base_local_path_model, subdirectories=subdirectories, download_file_patterns=download_model_patterns)
    else:
        print("Invalid action. Please enter 'upload' or 'download'.")

    sftp.close()
    ssh.close()
    print("Operation completed.")
except paramiko.SSHException as e:
    print(f"SSH error: {e}")
except Exception as e:
    print(f"An error occurred: {e}")