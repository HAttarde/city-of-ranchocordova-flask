"""
Groq API Client Module
======================

Provides a unified interface for Groq LLM API calls, replacing local model inference.
Supports both standard and streaming responses.
"""

import json
import os
from typing import Generator, Optional

from groq import Groq

# Default model - Llama 3.3 70B is free and very capable
DEFAULT_MODEL = "llama-3.3-70b-versatile"


class GroqClient:
    """
    Singleton Groq API client for the Rancho Cordova chatbot.
    """
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
            
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError(
                "GROQ_API_KEY environment variable not set. "
                "Get your free API key at https://console.groq.com"
            )
        
        self.client = Groq(api_key=api_key)
        self.model = os.getenv("GROQ_MODEL", DEFAULT_MODEL)
        self._initialized = True
        print(f"✅ Groq client initialized with model: {self.model}")
    
    def generate_response(
        self,
        system_message: str,
        user_message: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
    ) -> str:
        """
        Generate a standard (non-streaming) response.
        
        Args:
            system_message: System prompt defining assistant behavior
            user_message: User's query
            max_tokens: Maximum tokens in response
            temperature: Sampling temperature (0-1)
        
        Returns:
            Generated response text
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_message},
                ],
                max_tokens=max_tokens,
                temperature=temperature,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"❌ Groq API error: {e}")
            raise
    
    def generate_response_streaming(
        self,
        system_message: str,
        user_message: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
    ) -> Generator[str, None, None]:
        """
        Generate a streaming response (yields tokens as they arrive).
        
        Args:
            system_message: System prompt defining assistant behavior
            user_message: User's query
            max_tokens: Maximum tokens in response
            temperature: Sampling temperature (0-1)
        
        Yields:
            Token strings as they are generated
        """
        try:
            stream = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_message},
                ],
                max_tokens=max_tokens,
                temperature=temperature,
                stream=True,
            )
            
            for chunk in stream:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
                    
        except Exception as e:
            print(f"❌ Groq streaming error: {e}")
            yield f"[Error: {str(e)}]"
    
    def generate_json(
        self,
        system_message: str,
        user_message: str,
        max_tokens: int = 1024,
        temperature: float = 0.3,
    ) -> Optional[dict]:
        """
        Generate a JSON response (for structured outputs like visualization specs).
        
        Args:
            system_message: System prompt (should instruct JSON output)
            user_message: User's query
            max_tokens: Maximum tokens in response
            temperature: Lower temperature for more deterministic JSON
        
        Returns:
            Parsed JSON dict or None if parsing fails
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_message},
                ],
                max_tokens=max_tokens,
                temperature=temperature,
                response_format={"type": "json_object"},
            )
            
            content = response.choices[0].message.content.strip()
            return json.loads(content)
            
        except json.JSONDecodeError as e:
            print(f"⚠️ JSON parse error: {e}")
            return None
        except Exception as e:
            print(f"❌ Groq JSON API error: {e}")
            return None


# Singleton accessor
_groq_client = None


def get_groq_client() -> GroqClient:
    """Get or create the singleton Groq client."""
    global _groq_client
    if _groq_client is None:
        _groq_client = GroqClient()
    return _groq_client
