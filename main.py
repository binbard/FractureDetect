import streamlit as st
from PIL import Image
import numpy as np
from ultralytics import YOLO
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as ReportLabImage
from reportlab.lib.units import inch
import io
import random

model = YOLO('./models/best.pt')

st.title("Fracture Detection")

patient_id = st.text_input("Patient ID", value=str(random.randint(10000, 99999)))

doctor_name = st.text_input("Doctor Name", value=random.choice(["Dr. Smith", "Dr. Johnson", "Dr. Williams", "Dr. Brown", "Dr. Jones"]))

patient_name = st.text_input("Patient Name")
report_date = st.date_input("Report Date")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None and patient_name and patient_id and doctor_name:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', width=400)

    image_np = np.array(image)

    if st.button('Analyze'):
        with st.spinner('Detecting...'):
            results = model.predict(source=image_np, conf=0.25)

            for result in results:
                annotated_image = result.plot()
                annotated_pil_image = Image.fromarray(annotated_image)

                st.image(annotated_pil_image, caption='Detected Image', width=400)

                fracture_detected = 'fracture' in result.names.values()

                max_confidence = max(result.boxes.conf, default=0)
                if max_confidence < 0.50:
                    severity = "Low"
                elif 0.50 <= max_confidence < 0.80:
                    severity = "Medium"
                else:
                    severity = "Intense"

                fracture_status = "Fracture detected!" if fracture_detected else "No fracture detected."

                buffer = io.BytesIO()
                doc = SimpleDocTemplate(buffer, pagesize=letter)
                styles = getSampleStyleSheet()

                header_style = ParagraphStyle(name="HeaderStyle", fontSize=14, leading=16, spaceAfter=10, textColor=colors.darkblue)

                header_table_data = [
                    ["Patient Name:", patient_name],
                    ["Patient ID:", patient_id],
                    ["Doctor Name:", doctor_name],
                    ["Report Date:", report_date.strftime("%B %d, %Y")],
                ]
                header_table = Table(header_table_data, colWidths=[120, 300])
                header_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
                    ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, 0), 12),
                    ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                    ('BACKGROUND', (0, 1), (-1, -1), colors.whitesmoke),
                    ('GRID', (0, 0), (-1, -1), 1, colors.black),
                ]))

                title = Paragraph("Fracture Detection Report", styles['Title'])

                detection_table_data = [
                    ["Fracture Status", fracture_status],
                    ["Severity", severity],
                    ["Prediction Time", f"{sum(result.speed.values()):.2f} ms"],
                    ["Original Image Shape", f"{result.orig_shape}"],
                ]

                if result.boxes:
                    for i, box in enumerate(result.boxes.xyxy):
                        detection_table_data.append([f"Bounding Box {i + 1}", f"{box}"])

                detection_table = Table(detection_table_data, colWidths=[150, 300])
                detection_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
                    ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, 0), 12),
                    ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                    ('BACKGROUND', (0, 1), (-1, -1), colors.whitesmoke),
                    ('GRID', (0, 0), (-1, -1), 1, colors.black),
                ]))

                annotated_image_path = "./annotated_image.jpg"
                annotated_pil_image.save(annotated_image_path)
                report_image = ReportLabImage(annotated_image_path, width=5*inch, height=3.75*inch)

                elements = [title, Spacer(1, 12), header_table, Spacer(1, 12), detection_table, Spacer(1, 12), report_image]

                doc.build(elements)

                buffer.seek(0)
                st.download_button(
                    label="Download PDF Report",
                    data=buffer,
                    file_name=f"{patient_name}_fracture_detection_report.pdf",
                    mime="application/pdf"
                )
