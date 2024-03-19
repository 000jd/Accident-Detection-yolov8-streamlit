import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
from dotenv import load_dotenv
import os

class EmergencyEmailSender:
    def __init__(self):
        load_dotenv()
        self.sender_email = os.getenv("SENDER_EMAIL")
        self.receiver_email = os.getenv("RECEIVER_EMAIL")
        self.email_password = os.getenv("EMAIL_PASSWORD")
        self.subject = os.getenv("SUBJECT")

        # Check if any of the variables is missing
        if not (self.sender_email and self.receiver_email and self.email_password and self.subject):
            raise ValueError("Incomplete email configuration in the .env file")

    def send_emergency_email(self, video_name, timestamp, snapshot_path):
        body = f"Emergency detected in video: {video_name}\nTimestamp: {timestamp}"

        # Seting up the MIME
        message = MIMEMultipart()
        message["From"] = self.sender_email
        message["To"] = self.receiver_email
        message["Subject"] = self.subject
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
                server.login(self.sender_email, self.email_password)
                server.sendmail(self.sender_email, self.receiver_email, message.as_string())
            print("Emergency email sent successfully.")
        except Exception as e:
            print(f"Error sending emergency email: {e}")
