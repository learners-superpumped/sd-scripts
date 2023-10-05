import aiohttp
from typing import Dict, Any
import asyncio
from utils import convert_timestamp_to_datetime
import time
from utils.logger import logger

async def send_webhook(webhook_url: str, payload: Dict[str, Any]) -> bool:
    async with aiohttp.ClientSession() as session:
        async with session.post(webhook_url, json=payload) as response:
            if response.status == 200:
                logger.info("Webhook sent successfully")
                return True
            else:
                logger.error(f"Failed to send webhook. Status code: {response.status}")
                return False
    
def send_error_webhook(locon_inference_request, error, request_id, started_at):
    asyncio.run(
        send_webhook(
            locon_inference_request.webhook,
            payload={
                "id": request_id,
                "started_at": convert_timestamp_to_datetime(started_at),
                "completed_at": convert_timestamp_to_datetime(time.time()),
                "status": "failed",
                "error": str(error),
                "webhook": locon_inference_request.webhook,
            },
        )
    )
