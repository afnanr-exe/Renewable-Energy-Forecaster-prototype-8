import os
from azure.storage.blob import BlobServiceClient

def download_all_blobs_to_tmp(container_name: str) -> str:
    """
    Downloads ALL blobs from the given container into /tmp/aeso_raw/
    and returns the local directory path.
    """
    connection_string = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
    if not connection_string:
        raise RuntimeError("AZURE_STORAGE_CONNECTION_STRING is not set.")

    blob_service = BlobServiceClient.from_connection_string(connection_string)
    container = blob_service.get_container_client(container_name)

    # Create a folder inside /tmp for AESO files
    local_dir = "/tmp/aeso_raw"
    os.makedirs(local_dir, exist_ok=True)

    # Download every blob in the container
    for blob in container.list_blobs():
        local_path = os.path.join(local_dir, blob.name)
        with open(local_path, "wb") as f:
            f.write(container.get_blob_client(blob.name).download_blob().readall())

    return local_dir
