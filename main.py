import io
import uuid
import datetime
import os
from fastapi import FastAPI, File, UploadFile, Request, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from PIL import Image
import torch
import torchvision.transforms as transforms
import torch.nn as nn
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from torchvision.models import resnet50, ResNet50_Weights

app = FastAPI()

# Ensure temp directory exists
os.makedirs("temp", exist_ok=True)

# Mount static files and set templates directory
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Enable CORS (Allow all origins here, change for production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model
model_path = "diabetic_retinopathy_model.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = resnet50(weights=None)  # Avoids deprecated 'pretrained=True'
model.fc = nn.Linear(model.fc.in_features, 5)

state_dict = torch.load(model_path, map_location=device, weights_only=True)
model.load_state_dict(state_dict)

model.to(device)
model.eval()

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# Class labels for predictions
class_names = ["No DR", "Mild", "Moderate", "Severe", "Proliferative DR"]


@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        img_t = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(img_t)
            _, preds = torch.max(outputs, 1)
            pred_class = class_names[preds.item()]

        return JSONResponse(content={"prediction": pred_class})

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


@app.get("/download_report/")
async def download_report(
    result: str = Query(...),
    name: str = Query(...),
    age: str = Query(...),
    gender: str = Query(...)
):
    filename = f"report_{uuid.uuid4().hex}.pdf"
    filepath = f"temp/{filename}"

    c = canvas.Canvas(filepath, pagesize=A4)
    width, height = A4

    c.setFillColor(colors.HexColor("#002B5B"))
    c.setFont("Helvetica-Bold", 24)
    c.drawString(50, height - 60, "DRD - Diabetic Retinopathy Detection")

    c.setStrokeColor(colors.black)
    c.setLineWidth(1)
    c.line(50, height - 70, width - 50, height - 70)

    c.setFont("Helvetica-Bold", 16)
    c.setFillColor(colors.black)
    c.drawString(50, height - 110, "Patient Information")

    c.setFont("Helvetica", 12)
    y = height - 140
    c.drawString(60, y, f"Name     : {name}")
    c.drawString(300, y, f"Gender   : {gender}")
    y -= 20
    c.drawString(60, y, f"Age      : {age}")
    c.drawString(300, y, f"Date     : {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    y -= 40
    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, y, "Diagnosis Result:")

    y -= 20
    c.setFont("Helvetica", 12)
    if "severe" in result.lower() or "proliferative" in result.lower():
        c.setFillColor(colors.red)
    else:
        c.setFillColor(colors.black)
    c.drawString(60, y, result)

    # Recommendation Section
    y -= 40
    c.setFont("Helvetica", 11)
    c.setFillColor(colors.black)
    c.drawString(50, y, "Recommendation:")
    y -= 20

    result_lower = result.lower()
    if "no" in result_lower:
        recommendation = "No signs of Diabetic Retinopathy. Maintain a healthy lifestyle and get regular eye check-ups."
    elif "mild" in result_lower:
        recommendation = "Mild DR detected. Schedule a follow-up with an ophthalmologist within 6 months."
    elif "moderate" in result_lower:
        recommendation = "Moderate DR detected. Consult an ophthalmologist soon for further evaluation."
    elif "severe" in result_lower:
        recommendation = "Severe DR detected. Immediate consultation with an eye specialist is recommended."
    elif "proliferative" in result_lower:
        recommendation = "Proliferative DR detected. Urgent medical attention is required to prevent vision loss."
    else:
        recommendation = "Unrecognized result. Please consult a medical professional."

    c.drawString(60, y, recommendation)

    # Disclaimer
    c.setFont("Helvetica-Oblique", 9)
    c.setFillColor(colors.gray)
    c.drawString(50, 50, "Disclaimer: This is an AI-generated report. Please consult a doctor for further diagnosis.")

    c.save()

    return FileResponse(path=filepath, filename="DR_Report.pdf", media_type='application/pdf')
