# Blog Writer Agent

A sophisticated AI-powered blog content generator that creates SEO-optimized, plagiarism-checked blog posts using multiple personas and templates.

## Features

- 🤖 AI-powered content generation using OpenAI's GPT-4
- 🔍 SEO optimization and validation
- 📝 Multiple writing personas
- 📋 Various blog post templates
- 🔎 Keyword research and analysis
- ✅ Plagiarism checking
- 📊 FAQ generation
- 🎯 Meta title and description optimization
- 🌐 REST API with FastAPI

## Prerequisites

- Python 3.8 or higher
- OpenAI API key
- Google Custom Search API key
- Google Custom Search Engine ID
- Plagiarism API key

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd blog-writer-agent
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. Download NLTK data:
```python
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
```

4. Set up environment variables:
   - Copy `.env.example` to `.env`
   - Fill in your API keys in the `.env` file:
     ```
     OPENAI_API_KEY=your_openai_api_key
     GOOGLE_API_KEY=your_google_api_key
     GOOGLE_CSE_ID=your_google_cse_id
     PLAGIARISM_API_KEY=your_plagiarism_api_key
     ```

## How It Works

The blog writer agent consists of several specialized components that work together:

### 1. KeywordAgent
- Performs keyword research using Google Custom Search API
- Deduplicates keywords using NLP techniques
- Returns a list of relevant keywords with search volume and competition metrics

### 2. ContentRetrievalAgent
- Fetches top 3 relevant content pieces for the target keyword
- Uses Google Custom Search API to find authoritative sources
- Provides context for content generation

### 3. BlogDraftAgent
- Generates blog content using OpenAI's GPT-4
- Uses predefined personas and templates
- Ensures E-E-A-T (Experience, Expertise, Authoritativeness, Trustworthiness)
- Includes FAQ section generation

### 4. ValidationAgent
- Performs SEO validation checks:
  - Keyword presence in title
  - Meta description length
  - Keyword density in content
  - FAQ section presence
- Checks content for plagiarism
- Returns validation results

## Available Personas

1. **Oliver Moore**
   - British expat in Dubai
   - Focus: Grocery solutions for home comforts
   - Expertise: Experience living abroad, customs knowledge

2. **Amelia Fischer**
   - European expat and global traveler
   - Focus: Culinary diversity
   - Expertise: Cultural expertise, local producers

3. **Victoria Hayes**
   - Health and wellness expert
   - Focus: Nutrition and meal planning
   - Expertise: Certified nutritionist, health food retail

## Blog Templates

1. **How-to Post**
   - Structure:
     - H1: Main keyword
     - H2: Introduction
     - H2: Step-by-step instructions
     - H2: Conclusion
     - H2: FAQ

2. **List Post**
   - Structure:
     - H1: Main keyword
     - H2: Introduction
     - H2: Numbered list items
     - H2: Summary
     - H2: FAQ

## Usage

### Command Line Usage

1. Basic usage:
```python
from agents import BlogCreator

async def main():
    creator = BlogCreator()
    blog = await creator.create_blog(
        topic="your topic",
        persona="Victoria Hayes",
        template="How-to Post",
        word_count=1500
    )
    print(f"Blog generated successfully!\nTitle: {blog.meta_title}")
```

2. Custom parameters:
```python
blog = await creator.create_blog(
    topic="custom topic",
    persona="Oliver Moore",
    template="List Post",
    word_count=2000
)
```

### API Usage

The blog writer agent is also available as a REST API using FastAPI:

1. Start the API server:
```bash
python fastapi_agents.py
```

2. The API will be available at `http://localhost:8000`

3. API Documentation:
   - Swagger UI: `http://localhost:8000/docs`
   - ReDoc: `http://localhost:8000/redoc`

4. Generate a blog post using the API:
```bash
curl -X POST "http://localhost:8000/generate-blog" \
     -H "Content-Type: application/json" \
     -d '{
           "topic": "healthy grocery shopping",
           "persona": "Victoria Hayes",
           "template": "How-to Post",
           "word_count": 1500
         }'
```

5. Python client example:
```python
import requests

response = requests.post(
    "http://localhost:8000/generate-blog",
    json={
        "topic": "healthy grocery shopping",
        "persona": "Victoria Hayes",
        "template": "How-to Post",
        "word_count": 1500
    }
)

blog = response.json()
print(f"Blog title: {blog['meta_title']}")
print(f"Blog content: {blog['content'][:200]}...")
```

## Output Format

The blog generator returns a `BlogOutput` object containing:
- `content`: The generated blog post
- `meta_title`: SEO-optimized title
- `meta_description`: SEO-optimized description
- `faq`: List of frequently asked questions
- `plagiarism_score`: Plagiarism check percentage

## Error Handling

The system includes comprehensive error handling for:
- Missing API keys
- API rate limits
- Content generation failures
- SEO validation failures
- Plagiarism check failures

## Best Practices

1. **API Keys**
   - Keep your API keys secure
   - Use environment variables
   - Never commit keys to version control

2. **Content Generation**
   - Use appropriate personas for your topic
   - Select templates based on content type
   - Adjust word count based on topic complexity

3. **SEO Optimization**
   - Use relevant keywords
   - Keep meta descriptions under 160 characters
   - Include FAQ sections for better SEO

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details. 