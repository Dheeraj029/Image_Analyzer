import requests
import logging
from typing import Dict, Any, Optional
from openai import AzureOpenAI

# Configure Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HybridImageAnalyzer:
    """
    Combines Azure Computer Vision (Analysis) with Azure OpenAI (Synthesis).
    """

    def __init__(self, vision_endpoint, vision_key, openai_endpoint, openai_key, openai_deployment):
        # 1. Setup Computer Vision
        self.vision_url = f"{vision_endpoint.rstrip('/')}/vision/v3.2/analyze"
        self.vision_key = vision_key
        
        # 2. Setup Azure OpenAI
        self.openai_deployment = openai_deployment
        try:
            self.openai_client = AzureOpenAI(
                azure_endpoint=openai_endpoint,
                api_key=openai_key,
                api_version="2024-02-15-preview"
            )
        except Exception as e:
            logger.error(f"OpenAI Client Init Error: {e}")
            self.openai_client = None

    def analyze_visual_features(self, image_bytes: bytes) -> Dict[str, Any]:
        """
        Calls Azure Computer Vision v3.2 for Object Detection & Tagging.
        """
        headers = {
            'Ocp-Apim-Subscription-Key': self.vision_key,
            'Content-Type': 'application/octet-stream'
        }
        
        params = {
            'visualFeatures': 'Description,Objects,Tags,Color',
            'language': 'en'
        }

        try:
            response = requests.post(
                self.vision_url, 
                headers=headers, 
                params=params, 
                data=image_bytes, 
                timeout=10
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Vision API Error: {e}")
            return {"error": str(e)}

    def generate_human_summary(self, vision_result: Dict) -> Dict[str, str]:
        """
        Uses GPT to generate a Title and Summary.
        Returns: {'title': str, 'summary': str}
        """
        if not self.openai_client:
            return {"title": "Error", "summary": "OpenAI Client not initialized."}

        # Extract raw data to feed GPT
        description = vision_result.get("description", {}).get("captions", [{}])[0].get("text", "")
        tags = [t["name"] for t in vision_result.get("tags", []) if t["confidence"] > 0.6]
        objects = [o["object"] for o in vision_result.get("objects", [])]

        if not description:
            return {"title": "Unknown Image", "summary": "Not enough data to generate a summary."}

        # --- UPDATED PROMPT FOR TITLE + SUMMARY ---
        prompt = f"""
        Analyze this image data:
        - Basic Caption: "{description}"
        - Detected Tags: {', '.join(tags)}
        - Detected Objects: {', '.join(objects)}

        Perform two tasks:
        1. Generate a short, creative, catchy Title (max 6 words).
        2. Write a professional, 2-sentence summary.

        Return the result specifically in this format with a pipe separator:
        TITLE | SUMMARY
        """

        try:
            response = self.openai_client.chat.completions.create(
                model=self.openai_deployment,
                messages=[
                    {"role": "system", "content": "You are a creative AI assistant."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=150,
                temperature=0.7
            )
            
            content = response.choices[0].message.content.strip()
            
            # Parse the Title | Summary format
            if "|" in content:
                parts = content.split("|", 1)
                return {"title": parts[0].strip(), "summary": parts[1].strip()}
            else:
                # Fallback if AI forgets the pipe
                return {"title": "Image Analysis", "summary": content}

        except Exception as e:
            logger.error(f"OpenAI Generation Error: {e}")
            return {"title": "Error", "summary": "Could not generate AI summary."}

def make_decision(confidence: float) -> str:
    if confidence > 0.80: return "High Confidence"
    elif confidence > 0.50: return "Moderate Confidence"
    return "Uncertain Result"