"""
LLM Integration for Construction RAG.

Provides LLM functionality for:
- Generating chunk summaries
- Answering queries based on retrieved context
- Page title extraction

Uses OpenRouter API with GPT-4o-mini by default, but supports other providers.
"""

import os
from typing import List, Dict, Optional


# Default configuration
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_MODEL = "openai/gpt-4o-mini"


class OpenRouterLLM:
    """
    LLM client using OpenRouter API.
    
    OpenRouter provides a unified API for accessing multiple LLM providers
    including OpenAI, Anthropic, Google, and open-source models.
    
    Example:
        >>> llm = OpenRouterLLM()
        >>> summary = llm.generate_summary("Door schedule content...", "table")
        >>> print(summary)
    """
    
    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        api_key: Optional[str] = None,
        base_url: str = OPENROUTER_BASE_URL
    ):
        """
        Initialize OpenRouter client.
        
        Args:
            model: Model identifier (e.g., "openai/gpt-4o-mini", "anthropic/claude-3-haiku")
            api_key: OpenRouter API key. If not provided, reads from OPENROUTER_API_KEY env var.
            base_url: OpenRouter API base URL
        
        Raises:
            ValueError: If API key is not found
            ImportError: If openai package is not installed
        """
        self.model = model
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        
        if not self.api_key:
            raise ValueError(
                "OpenRouter API key not found. Please set OPENROUTER_API_KEY environment variable:\n"
                "  PowerShell: $env:OPENROUTER_API_KEY = 'your-key-here'\n"
                "  Bash: export OPENROUTER_API_KEY='your-key-here'\n"
                "  Or pass api_key parameter directly."
            )
        
        # Initialize OpenAI client with OpenRouter base URL
        try:
            from openai import OpenAI
            self.client = OpenAI(
                base_url=base_url,
                api_key=self.api_key
            )
        except ImportError:
            raise ImportError("Please install openai: pip install openai")
        
        self.total_tokens = 0
        self.total_calls = 0
    
    def generate(
        self,
        prompt: str,
        max_tokens: int = 200,
        temperature: float = 0.3
    ) -> str:
        """
        Generate a response from the LLM.
        
        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens in response
            temperature: Sampling temperature (0.0-2.0). Lower = more deterministic.
        
        Returns:
            Generated text response, or empty string on error
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temperature,
                extra_headers={
                    "HTTP-Referer": "https://github.com/construction-rag",
                    "X-Title": "Construction RAG Library"
                }
            )
            
            self.total_calls += 1
            if response.usage:
                self.total_tokens += response.usage.total_tokens
            
            return response.choices[0].message.content.strip()
        
        except Exception as e:
            print(f"LLM Error: {e}")
            return ""
    
    def generate_summary(self, content: str, chunk_type: str) -> str:
        """
        Generate a summary for a document chunk.
        
        Creates concise, human-readable summaries that help with retrieval
        by providing clean descriptions of noisy OCR content.
        
        Args:
            content: The chunk content to summarize
            chunk_type: Type of chunk ('text', 'table', 'viewport')
        
        Returns:
            Summary string
        """
        # Skip very short content
        if len(content) < 20:
            return content
        
        if chunk_type == "table":
            prompt = f"""Summarize this construction drawing table in 1-2 sentences.
Focus on: what data it contains, key values/quantities, purpose.

Table content:
{content[:1500]}

Summary:"""
        
        elif chunk_type == "viewport":
            prompt = f"""Summarize this construction drawing viewport/figure in 1 sentence.
Describe: what drawing type (plan, section, detail), what it shows.

Content:
{content[:500]}

Summary:"""
        
        else:  # text, title_block, notes
            prompt = f"""Summarize this construction document text in 1-2 sentences.
Focus on: key information, specifications, requirements.

Text:
{content[:1500]}

Summary:"""
        
        return self.generate(prompt, max_tokens=100, temperature=0.2)
    
    def answer_query(self, query: str, contexts: List[str]) -> str:
        """
        Generate an answer to a query based on retrieved context.
        
        Args:
            query: User question
            contexts: List of relevant document chunks
        
        Returns:
            Answer string grounded in the provided context
        """
        context_text = "\n\n---\n\n".join(contexts[:5])  # Limit to 5 chunks
        
        prompt = f"""You are answering questions about construction drawings and documents.
Use ONLY the provided context to answer. If the answer isn't in the context, say "Not found in documents."

Context:
{context_text[:3000]}

Question: {query}

Answer:"""
        
        return self.generate(prompt, max_tokens=300, temperature=0.3)
    
    def extract_page_title(self, chunk_summaries: List[str]) -> str:
        """
        Extract the page/sheet title from chunk summaries.
        
        Analyzes chunk summaries to determine the drawing sheet type
        (e.g., "Floor Plan", "Door Schedule", "General Notes").
        
        Args:
            chunk_summaries: List of chunk summary strings
        
        Returns:
            Extracted page title
        """
        summaries_text = "\n".join(chunk_summaries[:10])
        
        prompt = f"""Based on these chunk summaries from a construction drawing, 
identify the sheet/page type in 2-4 words (e.g., "Floor Plan", "Door Schedule", "General Notes", "Site Plan").

Summaries:
{summaries_text[:1500]}

Sheet type:"""
        
        return self.generate(prompt, max_tokens=20, temperature=0.1)
    
    def get_stats(self) -> Dict:
        """Get usage statistics."""
        return {
            "total_calls": self.total_calls,
            "total_tokens": self.total_tokens,
            "model": self.model
        }


def generate_summaries_for_chunks(
    chunks: List[Dict],
    llm: OpenRouterLLM,
    verbose: bool = True
) -> List[Dict]:
    """
    Generate summaries for all chunks.
    
    Args:
        chunks: List of chunk dictionaries
        llm: LLM client instance
        verbose: Whether to print progress
    
    Returns:
        Chunks with summaries added
    """
    if verbose:
        print(f"Generating summaries for {len(chunks)} chunks using {llm.model}...")
    
    for i, chunk in enumerate(chunks, 1):
        # Skip if already has summary
        if chunk.get("summary") and len(chunk["summary"]) > 10:
            continue
        
        content = chunk.get("content", "")
        chunk_type = chunk.get("chunk_type", "text")
        
        # Generate summary
        summary = llm.generate_summary(content, chunk_type)
        chunk["summary"] = summary
        
        # Progress update
        if verbose and (i % 10 == 0 or i == len(chunks)):
            print(f"  Progress: {i}/{len(chunks)} chunks processed")
    
    return chunks
