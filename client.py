import requests, os

def upload_image(url, image_path):
    """
    Uploads an image to the specified URL.

    Args:
    - url (str): The URL to which the image is uploaded.
    - image_path (str): The path to the image file to upload.
    
    Returns:
    - The response from the server.
    """
    # Open the file in binary mode
    with open(image_path, 'rb') as file:
        # Define the request headers for a multipart/form-data upload
        files = {'file': (os.path.basename(image_path), file)}
        # Make the POST request
        response = requests.post(url, files=files)
    
    return response

# Define the URL of your FastAPI server's endpoint
api_url = 'http://localhost:8000/upload-image/'

# Specify the path to your image file
image_path = '/home/ubuntu/gradio_tool/86041327_p0_resized.webp'

# Upload the image and print the response
response = upload_image(api_url, image_path)
if response.status_code == 200:
    print("Upload successful. Server response:")
    print(response.json())
else:
    print(f"Upload failed. Status code: {response.status_code}")
