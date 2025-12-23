import os
import time
from typing import Any

from azure.core.exceptions import ResourceNotFoundError
from azure.storage.blob import BlobServiceClient


def blob_exists(config, blob_name: str) -> bool:
    client = BlobServiceClient.from_connection_string(
        config.azure_storage.connection_string
    )
    container = client.get_container_client(config.azure_storage.container_name)

    try:
        container.get_blob_client(blob_name).get_blob_properties()
        return True
    except ResourceNotFoundError:
        return False


def upload_file_to_azure(file_path: str, config: Any) -> str:
    if not os.path.exists(file_path):
        raise FileNotFoundError(file_path)

    client = BlobServiceClient.from_connection_string(
        config.azure_storage.connection_string
    )
    container = client.get_container_client(config.azure_storage.container_name)

    blob_name = os.path.basename(file_path)

    with open(file_path, "rb") as f:
        container.upload_blob(
            name=blob_name,
            data=f,
            overwrite=True,
        )

    return (
        f"azure://{config.azure_storage.account_name}.blob.core.windows.net/"
        f"{container.container_name}/{blob_name}"
    )


if __name__ == "__main__":
    from config import VectorDBType, get_config

    config = get_config(VectorDBType.CHROMADB)

    path = "./active.pdf"  # ANY FILE

    start = time.time()
    azure_uri = upload_file_to_azure(path, config)
    end = time.time()

    print("Uploaded URI:", azure_uri)
    print(f"Time: {end - start:.2f}s")

    exists = blob_exists(config, os.path.basename(path))
    print("Exists:", exists)
