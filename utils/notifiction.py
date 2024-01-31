import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage
from email.mime.base import MIMEBase
from email import encoders
from dotenv import load_dotenv
import os

# Loading environment variables from .env file
load_dotenv()

def send_emergency_email(video_name, timestamp, snapshot_path):
    # Email configuration
    sender_email = os.getenv("SENDER_EMAIL")
    receiver_email = os.getenv("RECEIVER_EMAIL")
    email_password = os.getenv("EMAIL_PASSWORD")

    # Check if any of the variables is missing
    if not (sender_email and receiver_email and email_password):
        raise ValueError("Incomplete email configuration in the .env file")

    subject = os.getenv("SUBJECT")
    body = f"Emergency detected in video: {video_name}\nTimestamp: {timestamp}"

    # Seting up the MIME
    message = MIMEMultipart()
    message["From"] = sender_email
    message["To"] = receiver_email
    message["Subject"] = subject
    message.attach(MIMEText(body, "plain"))

    # Attaching the image
    attachment = open(snapshot_path, "rb")
    image_part = MIMEBase("application", "octet-stream")
    image_part.set_payload(attachment.read())
    encoders.encode_base64(image_part)
    image_part.add_header("Content-Disposition", f"attachment; filename=emergency_snapshot_{timestamp}.png")
    message.attach(image_part)

    # Connecting to the SMTP server and send the email
    try:
        with smtplib.SMTP("your_smtp_server.com", 587) as server:
            server.starttls()
            server.login(sender_email, "your_email_password")
            server.sendmail(sender_email, receiver_email, message.as_string())
        print("Emergency email sent successfully.")
    except Exception as e:
        print(f"Error sending emergency email: {e}")

