"""
Email Notifier for Trading Simulation
Sends summary emails after each trading cycle.
"""

import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

# Email configuration from environment
SMTP_HOST = os.getenv("SMTP_HOST", "smtp.gmail.com")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SMTP_USER = os.getenv("SMTP_USER", "")
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD", "")
EMAIL_FROM = os.getenv("EMAIL_FROM", "")
EMAIL_TO = os.getenv("EMAIL_TO", "")


def send_trading_summary(result: Dict[str, Any], executed: bool = False) -> bool:
    """
    Send email summary of trading cycle.
    
    Args:
        result: Result from LiveTrader.run_cycle()
        executed: Whether trades were executed
        
    Returns:
        True if email sent successfully
    """
    if not all([SMTP_USER, SMTP_PASSWORD, EMAIL_TO]):
        logger.warning("Email not configured. Set SMTP_USER, SMTP_PASSWORD, EMAIL_TO")
        return False
    
    try:
        portfolio = result.get('portfolio', {})
        sells = result.get('recommendations', {}).get('sell', [])
        buys = result.get('recommendations', {}).get('buy', [])
        trades = result.get('executed_trades', [])
        positions = result.get('positions', [])
        
        # Build email content
        subject = f"📊 Trading Update: ${portfolio.get('total_value', 0):,.2f} ({portfolio.get('total_return', 0):+.2%})"
        
        html = f"""
        <html>
        <head>
            <style>
                body {{ font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto; }}
                .header {{ background: #1a1a2e; color: white; padding: 20px; text-align: center; }}
                .section {{ padding: 15px; border-bottom: 1px solid #eee; }}
                .metric {{ display: inline-block; margin: 10px 20px; text-align: center; }}
                .value {{ font-size: 24px; font-weight: bold; }}
                .label {{ color: #666; font-size: 12px; }}
                .positive {{ color: #00c853; }}
                .negative {{ color: #ff1744; }}
                table {{ width: 100%; border-collapse: collapse; }}
                th, td {{ padding: 8px; text-align: left; border-bottom: 1px solid #eee; }}
                .buy {{ color: #00c853; }}
                .sell {{ color: #ff1744; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>📊 Stock Trader</h1>
                <p>{datetime.now().strftime('%Y-%m-%d %H:%M')} EST</p>
            </div>
            
            <div class="section">
                <h2>💰 Portfolio Summary</h2>
                <div class="metric">
                    <div class="value">${portfolio.get('total_value', 0):,.2f}</div>
                    <div class="label">Total Value</div>
                </div>
                <div class="metric">
                    <div class="value {'positive' if portfolio.get('total_return', 0) >= 0 else 'negative'}">{portfolio.get('total_return', 0):+.2%}</div>
                    <div class="label">Total Return</div>
                </div>
                <div class="metric">
                    <div class="value">${portfolio.get('cash', 0):,.2f}</div>
                    <div class="label">Cash</div>
                </div>
                <div class="metric">
                    <div class="value">{portfolio.get('positions_count', 0)}</div>
                    <div class="label">Positions</div>
                </div>
            </div>
        """
        
        # Executed trades
        if trades:
            html += """
            <div class="section">
                <h2>✅ Executed Trades</h2>
                <table>
                    <tr><th>Action</th><th>Ticker</th><th>Shares</th><th>Price</th><th>P&L</th></tr>
            """
            for t in trades:
                pnl = t.get('pnl', 0)
                pnl_str = f"${pnl:+,.2f}" if t['action'] == 'SELL' else "-"
                html += f"""
                    <tr>
                        <td class="{t['action'].lower()}">{t['action']}</td>
                        <td><strong>{t['ticker']}</strong></td>
                        <td>{t['shares']}</td>
                        <td>${t['price']:.2f}</td>
                        <td class="{'positive' if pnl > 0 else 'negative'}">{pnl_str}</td>
                    </tr>
                """
            html += "</table></div>"
        
        # Recommendations (if not executed)
        if not executed and (buys or sells):
            html += """
            <div class="section">
                <h2>📋 Recommendations (Not Executed)</h2>
            """
            if sells:
                html += "<h3 class='sell'>Sell Signals</h3><ul>"
                for r in sells:
                    html += f"<li><strong>{r['ticker']}</strong>: {r['reason']}</li>"
                html += "</ul>"
            if buys:
                html += "<h3 class='buy'>Buy Signals</h3><ul>"
                for r in buys:
                    html += f"<li><strong>{r['ticker']}</strong> ({r['confidence']:.0%}): {r['target_shares']} shares @ ${r['current_price']:.2f}</li>"
                html += "</ul>"
            html += "</div>"
        
        # Current positions
        if positions:
            html += """
            <div class="section">
                <h2>📈 Open Positions</h2>
                <table>
                    <tr><th>Ticker</th><th>Shares</th><th>Entry</th><th>Current</th><th>P&L</th></tr>
            """
            for p in positions:
                pnl = p.get('unrealized_pnl', 0)
                pnl_pct = p.get('unrealized_pnl_pct', 0)
                html += f"""
                    <tr>
                        <td><strong>{p['ticker']}</strong></td>
                        <td>{p['shares']}</td>
                        <td>${p['entry_price']:.2f}</td>
                        <td>${p.get('current_price', 0):.2f}</td>
                        <td class="{'positive' if pnl >= 0 else 'negative'}">${pnl:+,.2f} ({pnl_pct:+.1f}%)</td>
                    </tr>
                """
            html += "</table></div>"
        
        if not trades and not buys and not sells:
            html += """
            <div class="section">
                <p>No trade recommendations at this time. Market may be closed or conditions not favorable.</p>
            </div>
            """
        
        html += """
            <div class="section" style="text-align: center; color: #666; font-size: 12px;">
                <p>Stock Market Trader - Automated Trading Simulation</p>
            </div>
        </body>
        </html>
        """
        
        # Send email
        msg = MIMEMultipart('alternative')
        msg['Subject'] = subject
        msg['From'] = EMAIL_FROM or SMTP_USER
        msg['To'] = EMAIL_TO
        
        msg.attach(MIMEText(html, 'html'))
        
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
            server.starttls()
            server.login(SMTP_USER, SMTP_PASSWORD)
            server.send_message(msg)
        
        logger.info(f"Email sent to {EMAIL_TO}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to send email: {e}")
        return False
