import re
from typing import List, Dict

AGRI_KNOWLEDGE_BASE = [
    {
        "keywords": ["early blight", "tomato blight", "potato blight", "target spot", "alternaria"],
        "answer": "Early Blight (caused by Alternaria solani) presents as dark brown spots with concentric rings (target pattern) on older leaves. **Remedies:** 1) Apply Chlorothalonil or Copper Fungicide preventatively every 7-10 days during humid weather. 2) Remove lower leaves that touch the soil to prevent spore splash. 3) Maintain a 3-year crop rotation away from Solanaceous crops (tomatoes, potatoes, peppers, eggplants)."
    },
    {
        "keywords": ["late blight", "phytophthora", "water-soaked", "irish potato"],
        "answer": "Late Blight (caused by Phytophthora infestans) is a severe oomycete disease. It causes rapid water-soaked leaf lesions with white mold on undersides during high humidity (>85%). **Action Required:** Destroy infected crop debris immediately. Apply Metalaxyl + Mancozeb spray preventatively before rainy spells. Do not compost infected leaves."
    },
    {
        "keywords": ["aphid", "spider mite", "thrips", "whitefly", "pest control", "insecticide"],
        "answer": "For soft-bodied sucking pests (aphids, spider mites, whiteflies): **Organic Control:** Spray Neem Oil (5ml/L of water + 1ml liquid soap) during evening hours. **Biological:** Release predatory ladybugs or lacewings. **Chemical:** Use Imidacloprid or Abamectin for severe infestations. Always adhere to the Pre-Harvest Interval (PHI)."
    },
    {
        "keywords": ["fertilizer", "npk", "nitrogen", "phosphorus", "potassium", "soil nutrient"],
        "answer": "NPK Balance: **Nitrogen (N)** promotes leaf growth; **Phosphorus (P)** aids root development & blooming; **Potassium (K)** enhances fruit quality & drought tolerance. For leafy vegetables, use higher N (e.g., 20-10-10). For fruit crops during flowering/fruiting, boost P & K (e.g., 10-26-26 or 13-0-45)."
    },
    {
        "keywords": ["irrigation", "water stress", "wilting", "drip", "frequency"],
        "answer": "Watering Guidelines: Deep, infrequent watering encourages deep root growth. Drip irrigation saves up to 40% water and keeps foliage dry, preventing fungal leaf spots. If leaves wilt in early morning, the crop needs immediate watering. Mid-day wilting can be temporary heat response if soil is moist."
    },
    {
        "keywords": ["weed", "parthenium", "cyperus", "herbicide", "mulch"],
        "answer": "Weed Management: 1) **Mulching:** Apply 3-4 inches of organic straw or black plastic mulch to suppress 90% of weed germination. 2) **Pre-emergence:** Apply Pendimethalin within 3 days of sowing. 3) **Post-emergence:** Glyphosate (directed spray) for non-crop areas, or Quizalofop-p-ethyl for narrow-leaf weeds in broadleaf crops."
    },
    {
        "keywords": ["banana ripeness", "fruit storage", "post-harvest", "shelf life"],
        "answer": "Banana Post-Harvest: Store harvested green bananas at 13-15°C with 85-90% relative humidity. Ethylene gas (100 ppm for 24h) triggers uniform ripening. Avoid storing below 12°C to prevent chilling injury (gray discoloration)."
    }
]


class ChatbotService:
    def __init__(self):
        pass

    def get_response(self, user_message: str, history: List[Dict[str, str]] = None) -> dict:
        user_msg_lower = user_message.lower()

        # 1. Search knowledge base
        best_match = None
        highest_score = 0

        for item in AGRI_KNOWLEDGE_BASE:
            score = sum(1 for kw in item["keywords"] if kw in user_msg_lower)
            if score > highest_score:
                highest_score = score
                best_match = item["answer"]

        if best_match and highest_score > 0:
            reply = best_match
        else:
            # Fallback general agricultural expert response
            reply = (
                f"Thank you for your question regarding **'{user_message}'**.\n\n"
                "As an AI Agricultural Advisory Specialist, here are standard recommendations:\n"
                "1. **Scout Fields Regularly:** Inspect leaf undersides and stem joints weekly for early pest signs.\n"
                "2. **Soil Health:** Perform a soil test every 2 years to optimize N-P-K and micronutrient application.\n"
                "3. **Integrated Pest Management (IPM):** Combine sticky traps, beneficial insects, and targeted bio-pesticides before resorting to synthetic chemicals.\n\n"
                "Feel free to upload a leaf or crop photo in the Diagnosis tab for instant AI computer-vision analysis!"
            )

        return {
            "user_message": user_message,
            "bot_response": reply,
            "suggested_followups": [
                "How do I prevent fungal leaf spots?",
                "What is the ideal NPK ratio for tomatoes?",
                "How can I set up drip irrigation efficiently?",
                "What weather conditions trigger late blight?"
            ]
        }


chatbot_service = ChatbotService()
