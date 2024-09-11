import numpy as np
import uvicorn
import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.security.api_key import APIKeyHeader
import secrets
from io import BytesIO
from PIL import Image
from typing import Tuple
from fastapi.responses import HTMLResponse
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision import models, datasets, transforms
from pydantic import BaseModel

app = FastAPI()
security = HTTPBasic()

# credentials
API_KEY = "test"
key_header = APIKeyHeader(name="access_token", auto_error=True)


def authenticate(api_key: str = Depends(key_header)):
    if api_key != API_KEY:
        raise HTTPException(
            status_code=403, detail="Wrong credentials"
        )

# def authenticate(api_key: str = Depends(key_header)):
#     correct_key = secrets.compare_digest(api_key, API_KEY)
#     if not (correct_key):
#         raise HTTPException(
#             status_code=401,
#             detail="Incorrect credentials",
#             headers={"WWW-Authenticate": "Basic"}
#         )
#     return api_key


# preprocessing
data_transforms = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


model = models.mobilenet_v2(pretrained=False)

num_ftrs = model.classifier[1].in_features
model.classifier[1] = torch.nn.Linear(num_ftrs, 2)

model.load_state_dict(torch.load('best_model.pt'))
model.eval()


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model.to(device)

# load data
# data_dir = 'datasetv1'
# dataset = datasets.ImageFolder(data_dir)

class_names = ['beach', 'mountain']


def read_file(data) -> Tuple[Image.Image, Tuple[int, int]]:
    img = Image.open(BytesIO(data)).convert('RGB')
    img_resized = img.resize((256, 256), resample=Image.BICUBIC)
    return img_resized, img_resized.size


@app.post("/predict")
async def predict(file: UploadFile = File(...), api_key: str = Depends(authenticate)):
    try:
        img, _ = read_file(await file.read())

        image = data_transforms(img)
        img_batch = image.unsqueeze(0)

        img_batch = img_batch.to(device)

        with torch.no_grad():
            outputs = model(img_batch)
            _, predicted = torch.max(outputs, 1)
            predicted_class = class_names[predicted.item()]
            confidences = torch.softmax(outputs, dim=1)[0]

            confidence_dict = {
                class_names[i]: float(confidences[i])
                for i in range(len(class_names))
            }

        return {
            'class': predicted_class,
            'confidence': confidence_dict
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000)
