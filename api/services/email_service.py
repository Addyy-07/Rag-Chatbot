"""
api/services/email_service.py
─────────────────────────────
Async SMTP email sender for OTP verification emails.
Uses aiosmtplib for non-blocking email delivery.
Falls back to console logging when SMTP is not configured.
"""

import logging
import aiosmtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

from api.config import api_settings

log = logging.getLogger(__name__)


def _build_otp_html(otp_code: str) -> str:
    """Render a clean HTML email template for the OTP."""
    return f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
    </head>
    <body style="margin:0;padding:0;background-color:#f4f4f5;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;">
        <table width="100%" cellpadding="0" cellspacing="0" style="background-color:#f4f4f5;padding:40px 20px;">
            <tr>
                <td align="center">
                    <table width="480" cellpadding="0" cellspacing="0" style="background-color:#ffffff;border-radius:12px;overflow:hidden;box-shadow:0 1px 3px rgba(0,0,0,0.1);">
                        <!-- Header -->
                        <tr>
                            <td style="background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);padding:32px 40px;text-align:center;">
                                <h1 style="margin:0;color:#ffffff;font-size:24px;font-weight:700;">🧠 DocChat AI</h1>
                            </td>
                        </tr>
                        <!-- Body -->
                        <tr>
                            <td style="padding:40px;">
                                <h2 style="margin:0 0 16px;color:#0f172a;font-size:20px;font-weight:600;">Verify your email</h2>
                                <p style="margin:0 0 24px;color:#64748b;font-size:15px;line-height:1.6;">
                                    Use the verification code below to complete your signup. This code expires in <strong>10 minutes</strong>.
                                </p>
                                <!-- OTP Box -->
                                <div style="background-color:#f8fafc;border:2px dashed #cbd5e1;border-radius:8px;padding:24px;text-align:center;margin:0 0 24px;">
                                    <span style="font-size:36px;font-weight:700;letter-spacing:8px;color:#0f172a;font-family:'Courier New',monospace;">
                                        {otp_code}
                                    </span>
                                </div>
                                <p style="margin:0;color:#94a3b8;font-size:13px;line-height:1.5;">
                                    If you didn't request this code, you can safely ignore this email.
                                </p>
                            </td>
                        </tr>
                        <!-- Footer -->
                        <tr>
                            <td style="background-color:#f8fafc;padding:20px 40px;text-align:center;border-top:1px solid #e2e8f0;">
                                <p style="margin:0;color:#94a3b8;font-size:12px;">
                                    © 2026 DocChat AI. All rights reserved.
                                </p>
                            </td>
                        </tr>
                    </table>
                </td>
            </tr>
        </table>
    </body>
    </html>
    """


async def send_otp_email(to_email: str, otp_code: str) -> bool:
    """
    Send the OTP verification email.
    Returns True on success, False on failure.
    Falls back to console output if SMTP is not configured.
    """
    # Check if SMTP is configured
    if not api_settings.smtp_user or not api_settings.smtp_password:
        log.warning("SMTP not configured — printing OTP to console instead.")
        log.info("=" * 50)
        log.info(f"  📧 OTP EMAIL (console fallback)")
        log.info(f"  To:   {to_email}")
        log.info(f"  Code: {otp_code}")
        log.info("=" * 50)
        return True

    try:
        msg = MIMEMultipart("alternative")
        msg["Subject"] = f"Your DocChat AI verification code: {otp_code}"
        msg["From"] = api_settings.smtp_from_email
        msg["To"] = to_email

        # Plain text fallback
        text_part = MIMEText(
            f"Your DocChat AI verification code is: {otp_code}\n\n"
            f"This code expires in 10 minutes.\n"
            f"If you didn't request this, ignore this email.",
            "plain",
        )
        html_part = MIMEText(_build_otp_html(otp_code), "html")

        msg.attach(text_part)
        msg.attach(html_part)

        await aiosmtplib.send(
            msg,
            hostname=api_settings.smtp_host,
            port=api_settings.smtp_port,
            username=api_settings.smtp_user,
            password=api_settings.smtp_password,
            use_tls=True,  # SSL on port 465
        )

        log.info(f"OTP email sent to {to_email}")
        return True

    except Exception as e:
        log.error(f"Failed to send OTP email to {to_email}: {e}")
        return False
