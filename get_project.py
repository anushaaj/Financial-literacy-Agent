"""List IBM watsonx.ai projects using credentials from the environment."""

import os

from dotenv import load_dotenv
from ibm_watsonx_ai import APIClient, Credentials


load_dotenv()

api_key = os.getenv("WATSONX_APIKEY")
if not api_key:
    raise SystemExit("Set WATSONX_APIKEY in your .env file before running this script.")

credentials = Credentials(
    api_key=api_key,
    url=os.getenv("WATSONX_URL", "https://us-south.ml.cloud.ibm.com"),
)
client = APIClient(credentials)

for project in client.projects.get_list():
    metadata = project.get("metadata", {})
    print(f"Project Name: {metadata.get('name')} | Project ID: {metadata.get('id')}")
