from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import uvicorn
import aiofiles
import os
import faiss
import json
from clip import CLIPEmbedder
from PIL import Image
import httpx
import subprocess

app = FastAPI()

# Global variables to hold model and index
index = None
danrooru_idx_to_fn = {}
embedder = None

def setup_model_and_index():
    global index, danrooru_idx_to_fn, embedder
    
    index_path = 'danrooru_auto.index'
    if not os.path.exists(index_path):
        subprocess.run(['aws', 's3', 'cp', 's3://pixai-test-uw2/larry/danrooru_auto.index', index_path])
    
    index_to_fn_json = 'danbooru_auto.json'
    if not os.path.exists(index_to_fn_json):
        subprocess.run(['aws', 's3', 'cp', 's3://pixai-test-uw2/larry/danbooru_auto.json', index_to_fn_json])
    
    index = faiss.read_index(index_path)
    danrooru_idx_to_fn = json.load(open(index_to_fn_json, 'r'))
    
    embedder = CLIPEmbedder()

async def process_image(image):
    embedding = embedder([image])
    embedding = embedding.astype('float32').reshape(1, -1)
    lims, _, I = index.range_search(embedding, 0.97)
    res_json = []
    
    if len(I) == 0:
        return res_json
    filenames = [danrooru_idx_to_fn[i] for i in I]  # Ensure keys are strings
    ids = [filename.replace('image_', '').split('.')[0] for filename in filenames]
    json_urls = [f'https://wfhteqiwu1.execute-api.us-west-2.amazonaws.com/default/record/{id}' for id in ids]

    async with httpx.AsyncClient() as client:
        for url in json_urls:
            response = await client.get(url)
            if response.status_code == 200:
                data = response.json()
                res_json.append(data)
    return res_json

@app.post("/upload-image/")
async def create_upload_file(file: UploadFile = File(...)):
    try:
        # Save temporary image file
        temp_file_path = f"temp_{file.filename}"
        async with aiofiles.open(temp_file_path, 'wb') as out_file:
            content = await file.read()  # read file content
            await out_file.write(content)  # write to temp file
        
        image = Image.open(temp_file_path)
        result = await process_image(image)
        
        # Optionally, clean up by removing the temporary file
        os.remove(temp_file_path)
        
        return JSONResponse(content={"results": result})
    except Exception as e:
        return HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    setup_model_and_index()
    print("Model and index loaded. Starting server...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
