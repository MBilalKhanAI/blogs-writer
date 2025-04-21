import openai
import asyncio
import uuid
import json
import requests
from typing import Dict, List, Optional
from dataclasses import dataclass
from uuid import uuid4
import nltk
from nltk.tokenize import word_tokenize
from dotenv import load_dotenv
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Load environment variables from .env file
load_dotenv()

# Configuration - Load API keys from environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GOOGLE_CSE_ID = os.getenv("GOOGLE_CSE_ID")
PLAGIARISM_API_KEY = os.getenv("PLAGIARISM_API_KEY")

# Validate that all required API keys are present
if not all([OPENAI_API_KEY, GOOGLE_API_KEY, GOOGLE_CSE_ID, PLAGIARISM_API_KEY]):
    raise ValueError("One or more required API keys are missing in the .env file.")

openai.api_key = OPENAI_API_KEY

# Initialize FastAPI app
app = FastAPI(title="McGrocer AI Blog Creator", description="API for generating E-E-A-T compliant blog content.")

# Persona Definitions
PERSONAS = {
    "Oliver Moore": {
        "bio": "British expat in Dubai for 5 years, sharing grocery solutions for home comforts.",
        "why": "To help fellow expats find familiar products abroad.",
        "eeat": "Experience living abroad, knowledge of customs/duties."
    },
    "Amelia Fischer": {
        "bio": "European expat and global traveler, passionate about culinary diversity.",
        "why": "To showcase European goods for a global audience.",
        "eeat": "Cultural expertise, referencing local producers."
    },
    "Victoria Hayes": {
        "bio": "Health and wellness expert with 10 years of experience in nutrition and meal planning.",
        "why": "To help people make healthier grocery shopping choices.",
        "eeat": "Certified nutritionist, experience in health food retail."
    }
}

# Blog Templates
TEMPLATES = {
    "How-to Post": {
        "structure": ["H1: {keyword}", "H2: Introduction", "H2: Step 1", "H2: Step 2", "H2: Conclusion", "H2: FAQ"],
        "prompt": "Write a step-by-step guide on {keyword}, ensuring clarity and actionable advice."
    },
    "List Post": {
        "structure": ["H1: {keyword}", "H2: Introduction", "H2: Item 1", "H2: Item 2", "H2: Summary", "H2: FAQ"],
        "prompt": "Create a list post on {keyword}, enumerating key tips or items with detailed explanations."
    },
    # Add other templates similarly
}

# Pydantic Models for API
class BlogRequest(BaseModel):
    topic: str
    persona: str
    template: str
    word_count: int = 1500

class BlogResponse(BaseModel):
    content: str
    meta_title: str
    meta_description: str
    faq: List[Dict[str, str]]
    plagiarism_score: float

@dataclass
class BlogConfig:
    persona: str
    keyword: str
    template: str
    word_count: int = 1500

@dataclass
class BlogOutput:
    content: str
    meta_title: str
    meta_description: str
    faq: List[Dict[str, str]]
    plagiarism_score: float

class KeywordAgent:
    async def fetch_keywords(self, topic: str) -> List[Dict]:
        url = f"https://www.googleapis.com/customsearch/v1?key={GOOGLE_API_KEY}&cx={GOOGLE_CSE_ID}&q={topic}"
        try:
            response = requests.get(url)
            response.raise_for_status()
            data = response.json()
            keywords = [
                {"keyword": item["title"], "search_volume": 1000, "competition": 0.5, "cpc": 1.0}
                for item in data.get("items", [])[:5]
            ]
        except Exception as e:
            keywords = [{"keyword": topic, "search_volume": 1000, "competition": 0.5, "cpc": 1.0}]
        return self.deduplicate_keywords(keywords)

    def deduplicate_keywords(self, keywords: List[Dict]) -> List[Dict]:
        seen = set()
        unique_keywords = []
        for kw in keywords:
            tokens = tuple(sorted(word_tokenize(kw["keyword"].lower())))
            if tokens not in seen:
                seen.add(tokens)
                unique_keywords.append(kw)
        return unique_keywords

class ContentRetrievalAgent:
    async def fetch_top_3_content(self, keyword: str) -> List[Dict]:
        url = f"https://www.googleapis.com/customsearch/v1?key={GOOGLE_API_KEY}&cx={GOOGLE_CSE_ID}&q={keyword}"
        try:
            response = requests.get(url)
            response.raise_for_status()
            data = response.json()
            return [
                {"title": item["title"], "snippet": item["snippet"], "url": item["link"]}
                for item in data.get("items", [])[:3]
            ]
        except Exception:
            return []

class BlogDraftAgent:
    async def generate_draft(self, config: BlogConfig, top_3_content: List[Dict]) -> Dict:
        persona = PERSONAS[config.persona]
        template = TEMPLATES[config.template]
        prompt = f"""
        You are {config.persona}, {persona['bio']}. Write a {config.template.lower()} for McGrocer on '{config.keyword}' ({config.word_count} words).
        Follow this structure: {', '.join(template['structure'])}.
        Ensure E-E-A-T:
        - Who: Include a byline and short bio.
        - How: Mention AI assistance and personal experience.
        - Why: Emphasize {persona['why']}.
        - E-E-A-T: Highlight {persona['eeat']}.
        Use top 3 content for inspiration (do not copy): {json.dumps(top_3_content)}.
        Include a FAQ section based on related questions.
        Ensure people-first content, avoiding SEO manipulation.
        """
        client = openai.OpenAIDefault()
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=int(config.word_count / 0.75),  # Approximate token-to-word ratio
        )
        content = response.choices[0].message.content
        meta_title = f"{config.keyword} | McGrocer Blog"
        meta_description = f"Explore {config.keyword} with {config.persona}. {persona['why'][:100]}..."
        faq = self.extract_faq(content)
        return {
            "content": content,
            "meta_title": meta_title,
            "meta_description": meta_description,
            "faq": faq
        }

    def extract_faq(self, content: str) -> List[Dict[str, str]]:
        return [{"question": f"What is {content[:20]}?", "answer": "This is a placeholder answer."}]

class ValidationAgent:
    async def validate_seo(self, draft: Dict, keyword: str) -> bool:
        content = draft["content"].lower()
        keyword = keyword.lower()
        checks = [
            keyword in draft["meta_title"].lower(),
            len(draft["meta_description"]) <= 160,
            keyword in content,
            len(draft["faq"]) > 0
        ]
        return sum(checks) >= 3

    async def check_plagiarism(self, content: str) -> float:
        url = "https://api.plagiarism-checker.com/check"
        try:
            response = requests.post(
                url,
                headers={"Authorization": f"Bearer {PLAGIARISM_API_KEY}"},
                json={"text": content}
            )
            response.raise_for_status()
            return response.json().get("overlap_percentage", 0.0)
        except Exception:
            return 0.0

class BlogCreator:
    def __init__(self):
        self.keyword_agent = KeywordAgent()
        self.content_agent = ContentRetrievalAgent()
        self.draft_agent = BlogDraftAgent()
        self.validation_agent = ValidationAgent()

    async def create_blog(self, topic: str, persona: str, template: str, word_count: int = 1500) -> BlogOutput:
        if persona not in PERSONAS:
            raise ValueError(f"Invalid persona: {persona}")
        if template not in TEMPLATES:
            raise ValueError(f"Invalid template: {template}")

        keywords = await self.keyword_agent.fetch_keywords(topic)
        primary_keyword = keywords[0]["keyword"]

        top_3_content = await self.content_agent.fetch_top_3_content(primary_keyword)

        config = BlogConfig(persona=persona, keyword=primary_keyword, template=template, word_count=word_count)
        draft = await self.draft_agent.generate_draft(config, top_3_content)

        seo_valid = await self.validation_agent.validate_seo(draft, primary_keyword)
        plagiarism_score = await self.validation_agent.check_plagiarism(draft["content"])
        if not seo_valid or plagiarism_score > 5.0:
            raise ValueError(f"Validation failed: SEO={'invalid' if not seo_valid else 'valid'}, Plagiarism={plagiarism_score}%")

        return BlogOutput(
            content=draft["content"],
            meta_title=draft["meta_title"],
            meta_description=draft["meta_description"],
            faq=draft["faq"],
            plagiarism_score=plagiarism_score
        )

# FastAPI Endpoint
@app.post("/generate-blog", response_model=BlogResponse)
async def generate_blog(request: BlogRequest):
    try:
        creator = BlogCreator()
        blog_output = await creator.create_blog(
            topic=request.topic,
            persona=request.persona,
            template=request.template,
            word_count=request.word_count
        )
        return BlogResponse(**blog_output.__dict__)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail="Internal server error")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)